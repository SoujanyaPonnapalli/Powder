package main

// Per-request SLO without retries or reconnects (G3).
//
// A request gets SLO (default 5 s) to obtain a partition slot, a connection and its result. If the SLO passes, the
// request is reported lost at once and never retried; its query is ended with KILL QUERY over a separate admin pool,
// which TiDB honours within milliseconds even while a write retries a quorum-less Region in prewrite (measured
// locally: 5.0 s with the kill vs ~54 s without; infra/tidb/local_kill_check). The connection stays open and returns
// to the pool afterwards, so when quorum comes back the same connections serve requests again.
//
// Why not the alternatives (G3 attempts 4-6): cancelling the query closes the socket (MySQL cannot abandon a query on
// a live connection), so every lost request cost a reconnect while TiDB kept the abandoned statement and its token
// for ~40-70 s; leaving the query running without a kill held the connection for the whole outage, because neither
// max_execution_time (reads only) nor tikv-client.commit-timeout (commit phase only) stops a write in prewrite.
//
// Per-partition slot budgets (bulkheads) keep one dead partition from using up connections that healthy partitions
// need. Sized to (requests per second to the partition) x SLO, a request never queues behind its own partition's
// lost requests either, so a request is lost only when TiDB did not answer it within its own SLO.

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// errSLO marks a request that did not finish within its SLO; Classify turns it into timeout/client_deadline.
var errSLO = fmt.Errorf("no result within the request SLO: %w", context.DeadlineExceeded)

// ClientStats are the counters written to <out>.pool.jsonl once per second.
type ClientStats struct {
	Inflight  atomic.Int64 // requests not yet reported
	Abandoned atomic.Int64 // queries still running after their request was reported lost
	Kills     atomic.Int64 // KILL QUERY sent and acknowledged
	KillErrs  atomic.Int64 // KILL QUERY failed (the query is then bounded only by ServerWait)
	SlotFull  atomic.Int64 // requests lost while waiting for their partition's slot
	logged    atomic.Int64
}

// logFirst writes the first few unexpected errors (failed kills, failed connection-id lookups) to stderr.
func (s *ClientStats) logFirst(what string, err error) {
	if s.logged.Add(1) <= 20 {
		fmt.Fprintf(os.Stderr, "%s %s: %v\n", time.Now().UTC().Format("15:04:05.000"), what, err)
	}
}

type Client struct {
	DB, Admin  *sql.DB
	SLO        time.Duration
	ServerWait time.Duration      // hard cap on a query after a failed kill (the context then closes the socket)
	Slots      []chan struct{}    // per-partition budgets; nil = none
	Stats      *ClientStats
	ids        sync.Map           // driver connection -> TiDB connection id
}

// PartitionBudget is the slot count that lets a partition receiving perSecond requests never queue behind its own
// lost requests: each holds a slot for at most about one SLO (plus the kill round trip).
func PartitionBudget(perSecond float64, slo time.Duration) int {
	return int(math.Ceil(perSecond*slo.Seconds())) + 2
}

func NewSlots(partitions, budget int) []chan struct{} {
	s := make([]chan struct{}, partitions)
	for i := range s {
		s[i] = make(chan struct{}, budget)
	}
	return s
}

// connID returns the TiDB connection id of conn, asking the server once per physical connection.
func (c *Client) connID(ctx context.Context, conn *sql.Conn) (int64, error) {
	var key any
	if err := conn.Raw(func(dc any) error { key = dc; return nil }); err != nil {
		return 0, err
	}
	if id, ok := c.ids.Load(key); ok {
		return id.(int64), nil
	}
	var id int64
	if err := conn.QueryRowContext(ctx, "SELECT CONNECTION_ID()").Scan(&id); err != nil {
		return 0, err
	}
	c.ids.Store(key, id)
	return id, nil
}

// Do runs work for a request to partition (-1 = no budget) under the SLO. connMS is the time spent getting a slot and
// a connection (it is part of the request's latency).
func (c *Client) Do(partition int, work func(ctx context.Context, conn *sql.Conn) error) (connMS float64, err error) {
	start := time.Now()
	deadline := start.Add(c.SLO)
	actx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	ms := func() float64 { return float64(time.Since(start)) / 1e6 }

	var slot chan struct{}
	if c.Slots != nil && partition >= 0 {
		slot = c.Slots[partition%len(c.Slots)]
		select {
		case slot <- struct{}{}:
		case <-actx.Done():
			c.Stats.SlotFull.Add(1)
			return ms(), errSLO
		}
	}
	release := func() {
		if slot != nil {
			<-slot
		}
	}
	conn, err := c.DB.Conn(actx)
	if err != nil {
		release()
		if errors.Is(err, context.DeadlineExceeded) {
			err = errSLO
		}
		return ms(), err
	}
	id, err := c.connID(actx, conn)
	connMS = ms()
	if err != nil {
		c.Stats.logFirst("connection id", err)
		conn.Close()
		release()
		if errors.Is(err, context.DeadlineExceeded) {
			err = errSLO
		}
		return connMS, err
	}

	var mu sync.Mutex
	finished, killed := false, false
	killDone := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		qctx, qcancel := context.WithTimeout(context.Background(), c.ServerWait)
		e := work(qctx, conn)
		qcancel()
		mu.Lock()
		finished = true
		k := killed
		mu.Unlock()
		if k {
			// never hand the connection to another request before our KILL has landed, then run a harmless
			// statement so a kill that arrived after the query ended cannot interrupt the next request's query
			<-killDone
			dctx, dcancel := context.WithTimeout(context.Background(), c.SLO)
			conn.ExecContext(dctx, "DO 1")
			dcancel()
		}
		conn.Close()
		release()
		done <- e
	}()
	select {
	case err = <-done:
		return connMS, err
	case <-time.After(time.Until(deadline)):
	}
	mu.Lock()
	if finished { // the result arrived at the deadline: report it
		mu.Unlock()
		return connMS, <-done
	}
	killed = true
	mu.Unlock()
	c.Stats.Abandoned.Add(1)
	go func() {
		kctx, kcancel := context.WithTimeout(context.Background(), c.SLO)
		if _, ke := c.Admin.ExecContext(kctx, fmt.Sprintf("KILL QUERY %d", id)); ke != nil {
			c.Stats.KillErrs.Add(1)
			c.Stats.logFirst(fmt.Sprintf("KILL QUERY %d", id), ke)
		} else {
			c.Stats.Kills.Add(1)
		}
		kcancel()
		close(killDone)
	}()
	go func() { <-done; c.Stats.Abandoned.Add(-1) }()
	return connMS, errSLO
}
