package main

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A fake TiDB: "SLOW" statements run until KILL QUERY <id> (from any connection) interrupts them, like a write to a
// quorum-less Region; "FAST" ones return at once. closed counts connections the pool closed (a reconnect would).
type fakeServer struct {
	mu     sync.Mutex
	nextID int64
	conns  map[int64]*fakeConn
	closed atomic.Int64
	deaf   atomic.Bool // ignore KILL QUERY (a server whose kill does not work)
}

type fakeConn struct {
	srv  *fakeServer
	id   int64
	kill chan struct{}
}

var errInterrupted = errors.New("Error 1317: Query execution was interrupted")

func (s *fakeServer) Connect(context.Context) (driver.Conn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	c := &fakeConn{srv: s, id: s.nextID, kill: make(chan struct{}, 1)}
	if s.conns == nil {
		s.conns = map[int64]*fakeConn{}
	}
	s.conns[c.id] = c
	return c, nil
}
func (s *fakeServer) Driver() driver.Driver { return nil }

func (c *fakeConn) Prepare(string) (driver.Stmt, error) { return nil, errors.New("not supported") }
func (c *fakeConn) Close() error                        { c.srv.closed.Add(1); return nil }
func (c *fakeConn) Begin() (driver.Tx, error)           { return nil, errors.New("not supported") }

func (c *fakeConn) ExecContext(ctx context.Context, q string, _ []driver.NamedValue) (driver.Result, error) {
	switch {
	case strings.HasPrefix(q, "KILL QUERY "):
		var id int64
		fmt.Sscanf(q, "KILL QUERY %d", &id)
		c.srv.mu.Lock()
		target := c.srv.conns[id]
		c.srv.mu.Unlock()
		if target != nil && !c.srv.deaf.Load() {
			select {
			case target.kill <- struct{}{}:
			default:
			}
		}
		return driver.RowsAffected(0), nil
	case q == "DO 1":
		select { // a kill that arrived after the query ended is consumed here, as the client intends
		case <-c.kill:
			return nil, errInterrupted
		default:
		}
		return driver.RowsAffected(0), nil
	case q == "SLOW":
		select {
		case <-c.kill:
			return nil, errInterrupted
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return driver.RowsAffected(1), nil
}

type idRows struct {
	id   int64
	done bool
}

func (r *idRows) Columns() []string { return []string{"CONNECTION_ID()"} }
func (r *idRows) Close() error      { return nil }
func (r *idRows) Next(dest []driver.Value) error {
	if r.done {
		return io.EOF
	}
	r.done = true
	dest[0] = r.id
	return nil
}

func (c *fakeConn) QueryContext(_ context.Context, q string, _ []driver.NamedValue) (driver.Rows, error) {
	if q != "SELECT CONNECTION_ID()" {
		return nil, errors.New("unexpected query " + q)
	}
	return &idRows{id: c.id}, nil
}

func newFakeClient(srv *fakeServer, slo time.Duration, partitions, budget int) (*Client, *countingConnector) {
	cc := &countingConnector{inner: srv}
	db := sql.OpenDB(cc)
	db.SetMaxOpenConns(partitions * budget)
	db.SetMaxIdleConns(partitions * budget)
	return &Client{DB: db, Admin: sql.OpenDB(srv), SLO: slo, ServerWait: time.Minute,
		Slots: NewSlots(partitions, budget), Stats: &ClientStats{}}, cc
}

func run(q string) func(context.Context, *sql.Conn) error {
	return func(ctx context.Context, conn *sql.Conn) error { _, err := conn.ExecContext(ctx, q); return err }
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(2 * time.Second); time.Now().Before(end); time.Sleep(5 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestLostRequestIsKilledAndItsConnectionReused(t *testing.T) {
	srv := &fakeServer{}
	cl, cc := newFakeClient(srv, 50*time.Millisecond, 1, 4)
	start := time.Now()
	_, err := cl.Do(0, run("SLOW")) // a write to a quorum-less Region
	if !errors.Is(err, errSLO) || time.Since(start) > 150*time.Millisecond {
		t.Fatalf("want errSLO at the SLO, got %v after %v", err, time.Since(start))
	}
	if o := Classify(err, false); o.Outcome != OutcomeTimeout {
		t.Fatalf("a lost request must classify as timeout, got %+v", o)
	}
	// the KILL ends the query promptly; the connection is not closed but returned to the pool
	waitFor(t, "the kill to land", func() bool { return cl.Stats.Abandoned.Load() == 0 })
	if cl.Stats.Kills.Load() != 1 || cl.Stats.KillErrs.Load() != 0 {
		t.Fatalf("kills=%d kill_errs=%d", cl.Stats.Kills.Load(), cl.Stats.KillErrs.Load())
	}
	// quorum is back: the next request succeeds at once on the same connection, no reconnect
	if _, err := cl.Do(0, run("FAST")); err != nil {
		t.Fatalf("next request: %v", err)
	}
	if cc.dials.Load() != 1 || srv.closed.Load() != 0 {
		t.Fatalf("want one request connection, never closed: dials=%d closed=%d", cc.dials.Load(), srv.closed.Load())
	}
}

func TestAFullPartitionDoesNotBlockAnother(t *testing.T) {
	// the budget is the safety net for kills that do not work: lost queries then hold their slots until ServerWait
	srv := &fakeServer{}
	srv.deaf.Store(true)
	slo := 200 * time.Millisecond
	cl, _ := newFakeClient(srv, slo, 2, 2)
	cl.ServerWait = 600 * time.Millisecond
	var wg sync.WaitGroup
	errs := make([]error, 3)
	for i := 0; i < 3; i++ { // partition 0 is dead: 3 requests for a budget of 2
		wg.Add(1)
		go func(i int) { defer wg.Done(); _, errs[i] = cl.Do(0, run("SLOW")) }(i)
	}
	time.Sleep(20 * time.Millisecond)
	start := time.Now()
	if _, err := cl.Do(1, run("FAST")); err != nil || time.Since(start) > 50*time.Millisecond {
		t.Fatalf("a healthy partition must not wait for a dead one: err=%v after %v", err, time.Since(start))
	}
	wg.Wait()
	for i, err := range errs {
		if !errors.Is(err, errSLO) {
			t.Fatalf("dead partition request %d: want errSLO, got %v", i, err)
		}
	}
	if cl.Stats.SlotFull.Load() != 1 {
		t.Fatalf("the request beyond the budget is lost waiting for a slot: slot_full=%d", cl.Stats.SlotFull.Load())
	}
	// ServerWait frees the dead partition's slots, so it serves again as soon as the server does
	waitFor(t, "the dead partition's slots to free", func() bool { return cl.Stats.Abandoned.Load() == 0 })
	if _, err := cl.Do(0, run("FAST")); err != nil {
		t.Fatalf("partition 0 after recovery: %v", err)
	}
}

func TestPartitionBudgetCoversRateTimesSLO(t *testing.T) {
	if b := PartitionBudget(4, 5*time.Second); b != 22 { // self-test: 200 req/s over 50 partitions
		t.Fatalf("budget %d", b)
	}
	if b := PartitionBudget(1, 5*time.Second); b != 7 { // one probe per second per partition
		t.Fatalf("probe budget %d", b)
	}
}
