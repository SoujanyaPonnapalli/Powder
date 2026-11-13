// local_kill_check: with quorum lost (2 of 3 TiKV paused), how long does a write take to give up, does KILL QUERY end it,
// and is the connection reusable afterwards? Prints one JSON line per write.
//
//	local_kill_check -dsn 127.0.0.1:4000 -kill-after 5s -wait 90s
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/go-sql-driver/mysql"
)

func open(dsn string) *sql.DB {
	cfg := mysql.NewConfig()
	cfg.User, cfg.Net, cfg.Addr, cfg.DBName = "root", "tcp", dsn, "test"
	cfg.InterpolateParams = true
	c, err := mysql.NewConnector(cfg)
	if err != nil {
		panic(err)
	}
	return sql.OpenDB(c)
}

func main() {
	dsn := flag.String("dsn", "127.0.0.1:4000", "")
	killAfter := flag.Duration("kill-after", 5*time.Second, "")
	wait := flag.Duration("wait", 90*time.Second, "give up waiting for an unkilled write after this")
	mode := flag.String("mode", "write", "setup | write")
	flag.Parse()
	db := open(*dsn)
	defer db.Close()
	ctx := context.Background()
	if *mode == "setup" {
		for _, q := range []string{"CREATE TABLE IF NOT EXISTS test.kt (id BIGINT PRIMARY KEY, v BIGINT)",
			"REPLACE INTO test.kt VALUES (1,0),(2,0),(3,0)"} {
			if _, err := db.ExecContext(ctx, q); err != nil {
				panic(err)
			}
		}
		fmt.Println("setup ok")
		return
	}
	type result struct {
		Case      string  `json:"case"`
		Seconds   float64 `json:"seconds"`
		Err       string  `json:"err"`
		Reusable  string  `json:"conn_reusable_after"`
		KillSentS float64 `json:"kill_sent_s,omitempty"`
	}
	run := func(name string, row int, kill bool, out chan<- result) {
		conn, err := db.Conn(ctx)
		if err != nil {
			panic(err)
		}
		var id int64
		if err := conn.QueryRowContext(ctx, "SELECT CONNECTION_ID()").Scan(&id); err != nil {
			panic(err)
		}
		ready <- struct{}{}
		<-paused
		r := result{Case: name}
		if kill {
			go func() {
				time.Sleep(*killAfter)
				r.KillSentS = killAfter.Seconds()
				if _, err := db.ExecContext(ctx, fmt.Sprintf("KILL QUERY %d", id)); err != nil {
					fmt.Fprintln(os.Stderr, "kill:", err)
				}
			}()
		}
		start := time.Now()
		qctx, cancel := context.WithTimeout(ctx, *wait) // a safety net only; the cases measure the server
		_, err = conn.ExecContext(qctx, "UPDATE test.kt SET v = v + 1 WHERE id = ?", row)
		cancel()
		r.Seconds = time.Since(start).Seconds()
		if err != nil {
			r.Err = err.Error()
		}
		reusable := make(chan string, 1)
		go func() {
			<-resumed
			var one int
			rctx, rc := context.WithTimeout(ctx, 60*time.Second)
			defer rc()
			if err := conn.QueryRowContext(rctx, "SELECT 1").Scan(&one); err != nil {
				reusable <- "no: " + err.Error()
			} else {
				reusable <- "yes"
			}
		}()
		r.Reusable = <-reusable
		conn.Close()
		out <- r
	}
	out := make(chan result, 2)
	var wg sync.WaitGroup
	for i, c := range []struct {
		name string
		kill bool
	}{{"no-kill", false}, {"kill-query-at-SLO", true}} {
		wg.Add(1)
		go func(i int, name string, kill bool) { defer wg.Done(); run(name, i+1, kill, out) }(i, c.name, c.kill)
	}
	for i := 0; i < 2; i++ {
		<-ready
	}
	fmt.Println("READY") // the driver script pauses 2 TiKV now and writes "paused\n" to stdin
	var line string
	fmt.Scanln(&line)
	close(paused)
	fmt.Scanln(&line) // "resumed"
	close(resumed)
	wg.Wait()
	close(out)
	for r := range out {
		b, _ := json.Marshal(r)
		fmt.Println(string(b))
	}
}

var (
	ready   = make(chan struct{}, 2)
	paused  = make(chan struct{})
	resumed = make(chan struct{})
)
