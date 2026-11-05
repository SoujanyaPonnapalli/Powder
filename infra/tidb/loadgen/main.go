// powder-load: open-loop load generator and per-partition prober for the Powder TiDB experiments (exec #7).
//
//	powder-load prepare --dsn host:4000 --keys 1000000 --regions 50 [--preload-rows N]
//	powder-load load    --dsn host:4000 --region us-east-1 --rate 500 --duration 10m --out requests.jsonl.gz
//	powder-load probe   --dsn host:4000 --region us-east-1 --buckets 50 --interval 1s --duration 10m --out probes.jsonl.gz
//	powder-load version
//
// Open loop: requests are issued on a fixed (or time-varying) schedule, each with its own client-side deadline that
// includes any wait for a connection. A closed-loop client stops sending while the database is down and would
// under-count an outage; this one does not. Records follow the schema in record.go.
package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/go-sql-driver/mysql"
)

const version = "powder-load 1"

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "powder-load: "+format+"\n", a...)
	os.Exit(2)
}

func openDB(dsn string, conns int, reqTimeout time.Duration, txnMode string) (*sql.DB, *countingConnector) {
	cfg := mysql.NewConfig()
	cfg.User, cfg.Net, cfg.Addr, cfg.DBName = "root", "tcp", dsn, "powder"
	cfg.Timeout = 5 * time.Second
	cfg.InterpolateParams = true
	// Server-side lock waits must not outlive the client deadline (default 50 s): at G3 abandoned statements kept
	// queueing on row locks after the client gave up, and the backlog saturated TiDB long after the fault ended.
	// max_execution_time does the same for reads (TiDB applies it to SELECT only).
	cfg.Params = map[string]string{"innodb_lock_wait_timeout": fmt.Sprint(int(math.Ceil(reqTimeout.Seconds()))),
		"max_execution_time": fmt.Sprint(reqTimeout.Milliseconds())}
	if txnMode != "" {
		// optimistic (the default): a single-statement write takes no pessimistic lock first. At G3 a KILL QUERY
		// interrupted TiDB's own pessimistic-lock rollback ("pessimisticRollback failed: query interrupted"), the
		// orphaned lock outlived the kill, and a prober rewriting the same row every second kept hitting it.
		cfg.Params["tidb_txn_mode"] = "'" + txnMode + "'"
	}
	c, err := mysql.NewConnector(cfg)
	if err != nil {
		fatal("dsn: %v", err)
	}
	// A TCP read can block far past the request deadline when a node is paused; the context deadline cancels it.
	cc := &countingConnector{inner: c}
	db := sql.OpenDB(cc)
	db.SetMaxOpenConns(conns)
	db.SetMaxIdleConns(conns)
	db.SetConnMaxLifetime(0)
	return db, cc
}

// newClient opens the request pool (sized to the partition budgets) and a small admin pool for KILL QUERY.
func newClient(dsn string, partitions, budget int, slo, serverWait time.Duration, txnMode string) (*Client, *countingConnector) {
	db, cc := openDB(dsn, partitions*budget, slo, txnMode)
	admin, _ := openDB(dsn, 16, slo, "")
	return &Client{DB: db, Admin: admin, SLO: slo, ServerWait: serverWait, Slots: NewSlots(partitions, budget),
		Stats: &ClientStats{}}, cc
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: powder-load prepare|load|probe|version [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "version":
		sum := sha256.Sum256(classTableJSON)
		fmt.Printf("%s error_classes=%s\n", version, hex.EncodeToString(sum[:8]))
	case "prepare":
		runPrepare(os.Args[2:])
	case "load":
		runLoad(os.Args[2:])
	case "probe":
		runProbe(os.Args[2:])
	default:
		fatal("unknown command %q", os.Args[1])
	}
}

// ------------------------------------------------------------------ prepare
func runPrepare(args []string) {
	fs := flag.NewFlagSet("prepare", flag.ExitOnError)
	dsn := fs.String("dsn", "127.0.0.1:4000", "TiDB host:port")
	keys := fs.Int64("keys", 1_000_000, "row id space [0, keys)")
	regions := fs.Int("regions", 50, "pre-split the table into this many Regions (one Raft group each)")
	preload := fs.Int64("preload-rows", 0, "insert this many rows (spread over the key space) so Regions hold data")
	payload := fs.Int("payload-bytes", 256, "value size")
	fs.Parse(args)

	boot, err := sql.Open("mysql", "root@tcp("+*dsn+")/?interpolateParams=true")
	if err != nil {
		fatal("%v", err)
	}
	defer boot.Close()
	must := func(q string, a ...any) {
		if _, err := boot.Exec(q, a...); err != nil {
			fatal("%s: %v", q, err)
		}
	}
	must("CREATE DATABASE IF NOT EXISTS powder")
	must("CREATE TABLE IF NOT EXISTS powder.kv (id BIGINT NOT NULL PRIMARY KEY, v VARBINARY(1024), ts BIGINT)")
	var tableID int64
	if err := boot.QueryRow("SELECT TIDB_TABLE_ID FROM information_schema.tables WHERE table_schema='powder' AND table_name='kv'").Scan(&tableID); err != nil {
		fatal("table id: %v", err)
	}
	if *regions > 1 {
		must(fmt.Sprintf("SPLIT TABLE powder.kv BETWEEN (0) AND (%d) REGIONS %d", *keys, *regions))
	}
	if *preload > 0 {
		step := *keys / *preload
		if step < 1 {
			step = 1
		}
		val := make([]byte, *payload)
		var wg sync.WaitGroup
		work := make(chan int64, 64)
		for w := 0; w < 16; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for start := range work {
					q := "REPLACE INTO powder.kv (id, v, ts) VALUES "
					a := make([]any, 0, 1500)
					for i := int64(0); i < 500 && start+i < *preload; i++ {
						if i > 0 {
							q += ","
						}
						q += "(?,?,?)"
						a = append(a, (start+i)*step, val, int64(0))
					}
					if _, err := boot.Exec(q, a...); err != nil {
						fatal("preload: %v", err)
					}
				}
			}()
		}
		for s := int64(0); s < *preload; s += 500 {
			work <- s
		}
		close(work)
		wg.Wait()
	}
	out, _ := json.Marshal(map[string]any{"table_id": tableID, "keys": *keys, "regions": *regions, "preloaded": *preload})
	fmt.Println(string(out))
}

// ------------------------------------------------------------------ load
func waitUntil(startAt float64) time.Time {
	t := time.Now()
	if startAt > 0 {
		t = time.Unix(0, int64(startAt*1e9))
	}
	if d := time.Until(t); d > 0 {
		time.Sleep(d)
	}
	return t
}

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
}

func epoch(t time.Time) float64 { return float64(t.UnixNano()) / 1e9 }

func runLoad(args []string) {
	fs := flag.NewFlagSet("load", flag.ExitOnError)
	dsn := fs.String("dsn", "127.0.0.1:4000", "TiDB host:port")
	region := fs.String("region", "local", "client region label written to every record")
	keys := fs.Int64("keys", 1_000_000, "row id space")
	rate := fs.Float64("rate", 500, "requests per second (mean, for diurnal)")
	duration := fs.Duration("duration", 10*time.Minute, "run length")
	timeout := fs.Duration("timeout", 5*time.Second, "per-request SLO: connection + result; later = lost, never retried")
	serverWait := fs.Duration("server-wait", time.Minute, "cap on a lost request's query if its KILL QUERY fails")
	buckets := fs.Int("buckets", 50, "key partitions (as the prober and the analysis use them) for connection budgets")
	txnMode := fs.String("txn-mode", envOr("POWDER_TXN_MODE", "optimistic"), "tidb_txn_mode for writes: optimistic|pessimistic")
	perPartition := fs.Int("conns-per-partition", 0, "connection budget per partition (0 = rate/buckets x timeout + 2, capped by -max-conns)")
	maxConns := fs.Int("max-conns", 600, "cap on all partition budgets together: with the prober's 7 per partition this keeps one "+
		"TiDB below its 1000 tokens even if every partition is dead, so KILL QUERY can always run")
	keyDist := fs.String("key-dist", "uniform", "uniform|zipf|hot-range|shifting-hot-range")
	arrival := fs.String("arrival", "constant", "constant|diurnal")
	period := fs.Duration("period", 24*time.Hour, "diurnal period")
	readRatio := fs.Float64("read-ratio", 0.5, "fraction of reads")
	hotFraction := fs.Float64("hot-fraction", 0.8, "hot-range: share of requests to the hot window")
	hotWidth := fs.Float64("hot-width", 0.05, "hot-range: window width as a fraction of the key space")
	shiftEvery := fs.Duration("shift-every", 0, "shifting-hot-range: move the window this often")
	seed := fs.Int64("seed", 1, "RNG seed")
	startAt := fs.Float64("start-at", 0, "epoch seconds to start at (synchronises load generators)")
	out := fs.String("out", "requests.jsonl.gz", "output records (gzip JSONL)")
	maxInflight := fs.Int("max-inflight", 20000, "cap on concurrent requests; beyond it requests are recorded as loadgen_saturated (other)")
	logOkEvery := fs.Int("log-ok-every", 1, "record every Nth successful request with weight N (failures are always recorded)")
	payloadBytes := fs.Int("payload-bytes", 256, "write value size")
	fs.Parse(args)

	kg, err := NewKeyGen(*keyDist, *keys, *hotFraction, *hotWidth, *shiftEvery)
	if err != nil {
		fatal("%v", err)
	}
	rt, err := NewRate(*arrival, *rate, *period)
	if err != nil {
		fatal("%v", err)
	}
	w, err := NewWriter(*out, *region)
	if err != nil {
		fatal("%v", err)
	}
	budget := *perPartition
	if budget <= 0 {
		budget = min(PartitionBudget(*rate/float64(*buckets), *timeout), max(2, *maxConns / *buckets))
	}
	fmt.Fprintf(os.Stderr, "connection budget: %d per partition x %d partitions\n", budget, *buckets)
	cl, cc := newClient(*dsn, *buckets, budget, *timeout, *serverWait, *txnMode)
	defer cl.DB.Close()
	defer cl.Admin.Close()
	sigCtx, stop := signalContext()
	defer stop()
	statsCtx, stopStats := context.WithCancel(context.Background())
	statsDone := make(chan struct{})
	go func() { poolStats(statsCtx, *out+".pool.jsonl", cl.DB, cc, cl.Stats); close(statsDone) }()

	payload := make([]byte, *payloadBytes)
	rng := rand.New(rand.NewSource(*seed))
	sem := make(chan struct{}, *maxInflight)
	var wg sync.WaitGroup
	var okN, toN, cfN, otN, saturated, okSeq atomic.Int64
	var maxLag time.Duration

	t0 := waitUntil(*startAt)
	next := t0
	issued := int64(0)
	for {
		elapsed := next.Sub(t0)
		if elapsed >= *duration || sigCtx.Err() != nil {
			break
		}
		if d := time.Until(next); d > 0 {
			time.Sleep(d)
		} else if -d > maxLag {
			maxLag = -d
		}
		row := kg.Next(rng, elapsed)
		read := rng.Float64() < *readRatio
		sched := epoch(next)
		issued++
		select {
		case sem <- struct{}{}:
			wg.Add(1)
			go func(row int64, read bool, sched float64) {
				defer wg.Done()
				defer func() { <-sem }()
				cl.Stats.Inflight.Add(1)
				defer cl.Stats.Inflight.Add(-1)
				start := time.Now()
				op := "write"
				if read {
					op = "read"
				}
				part := int(row * int64(*buckets) / *keys)
				connMS, err := cl.Do(part, func(ctx context.Context, conn *sql.Conn) error {
					if read {
						var v []byte
						err := conn.QueryRowContext(ctx, "SELECT v FROM powder.kv WHERE id=?", row).Scan(&v)
						if errors.Is(err, sql.ErrNoRows) {
							return nil
						}
						return err
					}
					_, err := conn.ExecContext(ctx, "REPLACE INTO powder.kv (id, v, ts) VALUES (?,?,?)", row, payload, start.UnixMilli())
					return err
				})
				o := Classify(err, false)
				rec := Record{TS: sched, Row: row, Op: op, Outcome: o, LatencyMS: float64(time.Since(start)) / 1e6,
					ConnMS: connMS, Weight: 1}
				if err != nil {
					rec.Err = err.Error()
				}
				switch o.Outcome {
				case OutcomeOK:
					okN.Add(1)
					if *logOkEvery > 1 {
						if okSeq.Add(1)%int64(*logOkEvery) != 0 {
							return
						}
						rec.Weight = *logOkEvery
					}
				case OutcomeTimeout:
					toN.Add(1)
				case OutcomeConflict:
					cfN.Add(1)
				default:
					otN.Add(1)
				}
				w.Add(rec)
			}(row, read, sched)
		default:
			saturated.Add(1)
			otN.Add(1)
			w.Add(Record{TS: sched, Row: row, Op: "read", Weight: 1,
				Outcome: Outcome{OutcomeOther, "loadgen_saturated", 0, "client"}, Err: "max-inflight reached"})
		}
		next = next.Add(time.Duration(float64(time.Second) / rt(next.Sub(t0))))
	}
	wg.Wait()
	stopStats()
	<-statsDone
	n := w.Close()
	sum, _ := json.Marshal(map[string]any{"issued": issued, "records": n, "ok": okN.Load(), "timeout": toN.Load(),
		"conflict": cfN.Load(), "other": otN.Load(), "saturated": saturated.Load(),
		"max_scheduler_lag_ms": float64(maxLag) / 1e6, "start": epoch(t0), "end": epoch(time.Now())})
	fmt.Println(string(sum))
}

// ------------------------------------------------------------------ probe
var severity = map[string]int{OutcomeOK: 0, OutcomeConflict: 1, OutcomeOther: 2, OutcomeTimeout: 3}

func runProbe(args []string) {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)
	dsn := fs.String("dsn", "127.0.0.1:4000", "TiDB host:port")
	region := fs.String("region", "local", "client region label")
	keys := fs.Int64("keys", 1_000_000, "row id space")
	buckets := fs.Int("buckets", 50, "partitions to probe (one row in the middle of each key bucket)")
	interval := fs.Duration("interval", time.Second, "probe period")
	duration := fs.Duration("duration", 10*time.Minute, "run length")
	timeout := fs.Duration("timeout", 5*time.Second, "per-probe SLO: connection + result; later = lost, never retried")
	serverWait := fs.Duration("server-wait", time.Minute, "cap on a lost probe's queries if its KILL QUERY fails")
	rowOffset := fs.Int64("row-offset", 0, "added to each bucket's middle row, so concurrent probers never write the same row")
	txnMode := fs.String("txn-mode", envOr("POWDER_TXN_MODE", "optimistic"), "tidb_txn_mode for writes: optimistic|pessimistic")
	startAt := fs.Float64("start-at", 0, "epoch seconds to start at")
	out := fs.String("out", "probes.jsonl.gz", "output records")
	fs.Parse(args)

	w, err := NewWriter(*out, *region)
	if err != nil {
		fatal("%v", err)
	}
	cl, cc := newClient(*dsn, *buckets, PartitionBudget(1/interval.Seconds(), *timeout), *timeout, *serverWait, *txnMode)
	defer cl.DB.Close()
	defer cl.Admin.Close()
	sigCtx, stop := signalContext()
	defer stop()
	statsCtx, stopStats := context.WithCancel(context.Background())
	statsDone := make(chan struct{})
	go func() { poolStats(statsCtx, *out+".pool.jsonl", cl.DB, cc, cl.Stats); close(statsDone) }()
	var wg sync.WaitGroup
	t0 := waitUntil(*startAt)
	var probes atomic.Int64
	for k := 0; ; k++ {
		sched := t0.Add(time.Duration(k) * *interval)
		if sched.Sub(t0) >= *duration || sigCtx.Err() != nil {
			break
		}
		if d := time.Until(sched); d > 0 {
			time.Sleep(d)
		}
		for b := 0; b < *buckets; b++ {
			row := int64((float64(b)+0.5)*float64(*keys)/float64(*buckets)) + *rowOffset
			wg.Add(1)
			probes.Add(1)
			go func(b int, row int64, sched float64) {
				defer wg.Done()
				cl.Stats.Inflight.Add(1)
				defer cl.Stats.Inflight.Add(-1)
				start := time.Now()
				connMS, err := cl.Do(b, func(ctx context.Context, conn *sql.Conn) error {
					if _, err := conn.ExecContext(ctx, "REPLACE INTO powder.kv (id, v, ts) VALUES (?,?,?)", row, []byte("probe"), start.UnixMilli()); err != nil {
						return err
					}
					var v []byte
					if err := conn.QueryRowContext(ctx, "SELECT v FROM powder.kv WHERE id=?", row).Scan(&v); err != nil && !errors.Is(err, sql.ErrNoRows) {
						return err
					}
					return nil
				})
				worst, errText := Classify(err, false), ""
				if err != nil {
					errText = err.Error()
				}
				w.Add(Record{TS: sched, Row: row, Op: "probe", Outcome: worst, LatencyMS: float64(time.Since(start)) / 1e6,
					ConnMS: connMS, Err: errText, Weight: 1})
			}(b, row, epoch(sched))
		}
	}
	wg.Wait()
	stopStats()
	<-statsDone
	n := w.Close()
	sum, _ := json.Marshal(map[string]any{"probes": probes.Load(), "records": n, "start": epoch(t0), "end": epoch(time.Now())})
	fmt.Println(string(sum))
}
