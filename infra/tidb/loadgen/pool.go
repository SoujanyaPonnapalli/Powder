package main

// Client-side instrumentation (G3 follow-up): at G3, healthy-Region requests timed out by the thousand while TiDB's
// slow log showed them fast, so the time must have gone to the client: waiting for a pool connection or setting one
// up. Every request records conn_ms (time to get a connection); a once-per-second pool snapshot records what
// database/sql and the connector see.
//
// <out>.pool.jsonl, one line per second:
//   ts, open, in_use, idle, wait_count, wait_ms (cumulative), dials, dial_errs, dial_ms (cumulative), inflight,
//   closed_idle, closed_lifetime

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"os"
	"sync/atomic"
	"time"
)

// countingConnector counts new connections (each costs TCP + MySQL handshake + session SETs) and their setup time.
type countingConnector struct {
	inner   driver.Connector
	dials   atomic.Int64
	errs    atomic.Int64
	totalNs atomic.Int64
}

func (c *countingConnector) Connect(ctx context.Context) (driver.Conn, error) {
	start := time.Now()
	conn, err := c.inner.Connect(ctx)
	c.dials.Add(1)
	c.totalNs.Add(int64(time.Since(start)))
	if err != nil {
		c.errs.Add(1)
	}
	return conn, err
}

func (c *countingConnector) Driver() driver.Driver { return c.inner.Driver() }

// poolStats writes one snapshot per second to path until ctx is done (and a last one then).
func poolStats(ctx context.Context, path string, db *sql.DB, cc *countingConnector, inflight *atomic.Int64) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pool stats: %v\n", err)
		return
	}
	defer f.Close()
	write := func() {
		s := db.Stats()
		fmt.Fprintf(f, `{"ts":%.3f,"open":%d,"in_use":%d,"idle":%d,"wait_count":%d,"wait_ms":%.1f,"dials":%d,`+
			`"dial_errs":%d,"dial_ms":%.1f,"inflight":%d,"closed_idle":%d,"closed_lifetime":%d}`+"\n",
			epoch(time.Now()), s.OpenConnections, s.InUse, s.Idle, s.WaitCount, float64(s.WaitDuration)/1e6,
			cc.dials.Load(), cc.errs.Load(), float64(cc.totalNs.Load())/1e6, inflight.Load(), s.MaxIdleClosed,
			s.MaxLifetimeClosed)
	}
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			write()
			return
		case <-tick.C:
			write()
		}
	}
}
