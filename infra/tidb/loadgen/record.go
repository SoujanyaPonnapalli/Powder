package main

// Per-request JSONL records (gzip-compressed, flushed every second so a crash loses at most one second).
// Schema (read by powder/deploy/availability.py):
//   ts            float epoch seconds: the SCHEDULED issue time (open loop: not delayed by a slow database)
//   client_region string, row int, op "read"|"write"|"probe"
//   outcome       ok|timeout|conflict|other ; cause, code, layer for every non-ok request ; latency_ms
//   err           first 200 chars of the error text (non-ok only)
//   w             weight (>1 when ok requests are down-sampled with --log-ok-every)

import (
	"bufio"
	"compress/gzip"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Record struct {
	TS        float64
	Row       int64
	Op        string
	Outcome   Outcome
	LatencyMS float64
	Err       string
	Weight    int
}

type Writer struct {
	mu   sync.Mutex
	f    *os.File
	gz   *gzip.Writer
	bw   *bufio.Writer
	ch   chan Record
	done chan struct{}
	reg  string
	n    int64
}

func NewWriter(path, clientRegion string) (*Writer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	gz := gzip.NewWriter(f)
	w := &Writer{f: f, gz: gz, bw: bufio.NewWriterSize(gz, 1<<20), ch: make(chan Record, 65536),
		done: make(chan struct{}), reg: clientRegion}
	go w.loop()
	return w, nil
}

func (w *Writer) Add(r Record) { w.ch <- r }

func (w *Writer) loop() {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case r, ok := <-w.ch:
			if !ok {
				w.flush()
				close(w.done)
				return
			}
			w.write(r)
		case <-tick.C:
			w.flush()
		}
	}
}

func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"' || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r < 0x20:
			b.WriteString(`\u00`)
			b.WriteString(strconv.FormatInt(int64(r)+0x100, 16)[1:])
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

func (w *Writer) write(r Record) {
	var b strings.Builder
	b.WriteString(`{"ts":`)
	b.WriteString(strconv.FormatFloat(r.TS, 'f', 6, 64))
	b.WriteString(`,"client_region":`)
	b.WriteString(jsonString(w.reg))
	b.WriteString(`,"row":`)
	b.WriteString(strconv.FormatInt(r.Row, 10))
	b.WriteString(`,"op":`)
	b.WriteString(jsonString(r.Op))
	b.WriteString(`,"outcome":`)
	b.WriteString(jsonString(r.Outcome.Outcome))
	if r.Outcome.Outcome != OutcomeOK {
		b.WriteString(`,"cause":`)
		b.WriteString(jsonString(r.Outcome.Cause))
		if r.Outcome.Code != 0 {
			b.WriteString(`,"code":`)
			b.WriteString(strconv.Itoa(r.Outcome.Code))
		}
		if r.Outcome.Layer != "" {
			b.WriteString(`,"layer":`)
			b.WriteString(jsonString(r.Outcome.Layer))
		}
		if r.Err != "" {
			b.WriteString(`,"err":`)
			b.WriteString(jsonString(truncate(r.Err, 200)))
		}
	}
	b.WriteString(`,"latency_ms":`)
	b.WriteString(strconv.FormatFloat(r.LatencyMS, 'f', 3, 64))
	if r.Weight > 1 {
		b.WriteString(`,"w":`)
		b.WriteString(strconv.Itoa(r.Weight))
	}
	b.WriteString("}\n")
	w.bw.WriteString(b.String())
	w.n++
}

func truncate(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func (w *Writer) flush() {
	w.bw.Flush()
	w.gz.Flush()
}

// Close drains the queue, finishes the gzip stream and returns the number of records written.
func (w *Writer) Close() int64 {
	close(w.ch)
	<-w.done
	w.gz.Close()
	w.f.Close()
	return w.n
}
