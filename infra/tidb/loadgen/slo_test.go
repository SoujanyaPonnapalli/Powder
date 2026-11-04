package main

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// fakeConn answers every Exec after *delay; closed counts connections the pool closed (a cancel would close one).
type fakeConn struct {
	delay  *atomic.Int64
	closed *atomic.Int64
}

func (c *fakeConn) Prepare(string) (driver.Stmt, error) { return nil, errors.New("not supported") }
func (c *fakeConn) Close() error                        { c.closed.Add(1); return nil }
func (c *fakeConn) Begin() (driver.Tx, error)           { return nil, errors.New("not supported") }
func (c *fakeConn) ExecContext(ctx context.Context, _ string, _ []driver.NamedValue) (driver.Result, error) {
	select {
	case <-time.After(time.Duration(c.delay.Load())):
		return driver.RowsAffected(1), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

type fakeConnector struct{ delay, closed atomic.Int64 }

func (f *fakeConnector) Connect(context.Context) (driver.Conn, error) {
	return &fakeConn{delay: &f.delay, closed: &f.closed}, nil
}
func (f *fakeConnector) Driver() driver.Driver { return nil }

func TestLostRequestKeepsItsConnectionAndTheNextRequestReusesIt(t *testing.T) {
	fc := &fakeConnector{}
	cc := &countingConnector{inner: fc}
	db := sql.OpenDB(cc)
	db.SetMaxOpenConns(1)
	defer db.Close()
	exec := func(ctx context.Context, conn *sql.Conn) error {
		_, err := conn.ExecContext(ctx, "REPLACE")
		return err
	}
	var abandoned atomic.Int64

	// a dead Region: the server answers only after 300 ms, the SLO is 50 ms
	fc.delay.Store(int64(300 * time.Millisecond))
	start := time.Now()
	_, err := withSLO(db, 50*time.Millisecond, time.Minute, &abandoned, exec)
	if !errors.Is(err, errSLO) || time.Since(start) > 150*time.Millisecond {
		t.Fatalf("want errSLO at the SLO, got %v after %v", err, time.Since(start))
	}
	if o := Classify(err, false); o.Outcome != OutcomeTimeout {
		t.Fatalf("a lost request must classify as timeout, got %+v", o)
	}
	if abandoned.Load() != 1 {
		t.Fatalf("the query should still be running: abandoned=%d", abandoned.Load())
	}

	// quorum is back: the next request waits for the same connection, which was never closed
	fc.delay.Store(0)
	time.Sleep(350 * time.Millisecond)
	if _, err := withSLO(db, time.Second, time.Minute, &abandoned, exec); err != nil {
		t.Fatalf("next request: %v", err)
	}
	if fc.closed.Load() != 0 || cc.dials.Load() != 1 || abandoned.Load() != 0 {
		t.Fatalf("want one connection, never closed: closed=%d dials=%d abandoned=%d",
			fc.closed.Load(), cc.dials.Load(), abandoned.Load())
	}
}

func TestWaitingForAConnectionCountsTowardsTheSLO(t *testing.T) {
	fc := &fakeConnector{}
	db := sql.OpenDB(&countingConnector{inner: fc})
	db.SetMaxOpenConns(1)
	defer db.Close()
	fc.delay.Store(int64(300 * time.Millisecond))
	var abandoned atomic.Int64
	exec := func(ctx context.Context, conn *sql.Conn) error { _, err := conn.ExecContext(ctx, "x"); return err }
	go withSLO(db, time.Second, time.Minute, &abandoned, exec) // holds the only connection for 300 ms
	time.Sleep(20 * time.Millisecond)
	start := time.Now()
	_, err := withSLO(db, 50*time.Millisecond, time.Minute, &abandoned, exec)
	if !errors.Is(err, errSLO) || time.Since(start) > 150*time.Millisecond {
		t.Fatalf("want errSLO while waiting for a connection, got %v after %v", err, time.Since(start))
	}
}
