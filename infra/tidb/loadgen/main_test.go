package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-sql-driver/mysql"
)

type vector struct {
	Code     int     `json:"code"`
	Text     string  `json:"text"`
	Deadline bool    `json:"deadline"`
	Outcome  string  `json:"outcome"`
	Cause    string  `json:"cause"`
	Layer    *string `json:"layer"`
}

// The same vectors drive the Python tests (tests/test_deploy_outcomes.py): the two classifiers must agree.
func TestClassifierMatchesSharedVectors(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "tests", "fixtures", "error_class_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Vectors []vector `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Vectors) < 15 {
		t.Fatalf("only %d vectors", len(doc.Vectors))
	}
	for _, v := range doc.Vectors {
		var err error
		if v.Code != 0 {
			err = &mysql.MySQLError{Number: uint16(v.Code), Message: v.Text}
		} else {
			err = errors.New(v.Text)
		}
		got := Classify(err, v.Deadline)
		wantLayer := ""
		if v.Layer != nil {
			wantLayer = *v.Layer
		}
		if got.Outcome != v.Outcome || got.Cause != v.Cause || got.Layer != wantLayer {
			t.Errorf("code=%d text=%q deadline=%v: got %+v, want %s/%s/%s", v.Code, v.Text, v.Deadline, got, v.Outcome, v.Cause, wantLayer)
		}
	}
}

func TestClassifySuccessAndContextErrors(t *testing.T) {
	if o := Classify(nil, false); o.Outcome != OutcomeOK || o.Cause != "" {
		t.Errorf("success: %+v", o)
	}
	if o := Classify(context.DeadlineExceeded, false); o.Cause != "client_deadline" {
		t.Errorf("context deadline: %+v", o)
	}
	if o := Classify(errors.New("anything"), true); o.Outcome != OutcomeTimeout {
		t.Errorf("deadline flag: %+v", o)
	}
}

func TestKeyDistributionsStayInRangeAndHaveTheirShape(t *testing.T) {
	const keys = 100000
	rng := rand.New(rand.NewSource(1))
	for _, kind := range []string{"uniform", "zipf", "hot-range", "shifting-hot-range"} {
		kg, err := NewKeyGen(kind, keys, 0.8, 0.05, time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20000; i++ {
			if r := kg.Next(rng, time.Duration(i)*time.Second); r < 0 || r >= keys {
				t.Fatalf("%s: row %d out of range", kind, r)
			}
		}
	}
	hot, _ := NewKeyGen("hot-range", keys, 0.8, 0.05, 0)
	in := 0
	for i := 0; i < 50000; i++ {
		if hot.Next(rng, 0) < keys*5/100 {
			in++
		}
	}
	if f := float64(in) / 50000; f < 0.78 || f > 0.86 { // 80% hot + 20%*5% uniform spill
		t.Errorf("hot-range fraction %.3f", f)
	}
	sh, _ := NewKeyGen("shifting-hot-range", keys, 1.0, 0.05, time.Minute)
	a, b := sh.Next(rng, 0), sh.Next(rng, 3*time.Minute)
	if a >= 5000 || b < 15000 || b >= 20000 { // window moves by one width per minute
		t.Errorf("shifting window: first=%d later=%d", a, b)
	}
	z, _ := NewKeyGen("zipf", keys, 0, 0, 0)
	low := 0
	for i := 0; i < 20000; i++ {
		if z.Next(rng, 0) < keys/100 {
			low++
		}
	}
	if float64(low)/20000 < 0.3 {
		t.Errorf("zipf not skewed: %d/20000 in the first 1%%", low)
	}
	if _, err := NewKeyGen("nope", keys, 0, 0, 0); err == nil {
		t.Error("unknown distribution accepted")
	}
	if _, err := NewKeyGen("shifting-hot-range", keys, 0.8, 0.05, 0); err == nil {
		t.Error("shifting without --shift-every accepted")
	}
}

func TestRateProfiles(t *testing.T) {
	c, _ := NewRate("constant", 500, 0)
	if c(0) != 500 || c(time.Hour) != 500 {
		t.Error("constant rate varies")
	}
	d, err := NewRate("diurnal", 1000, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	minR, maxR, sum, n := math.Inf(1), 0.0, 0.0, 0
	for m := 0; m < 24*60; m++ {
		r := d(time.Duration(m) * time.Minute)
		minR, maxR, sum, n = math.Min(minR, r), math.Max(maxR, r), sum+r, n+1
	}
	if math.Abs(minR-400) > 5 || math.Abs(maxR-1600) > 5 || math.Abs(sum/float64(n)-1000) > 2 {
		t.Errorf("diurnal min=%.0f max=%.0f mean=%.1f", minR, maxR, sum/float64(n))
	}
	if _, err := NewRate("diurnal", 100, 0); err == nil {
		t.Error("diurnal without period accepted")
	}
	if _, err := NewRate("constant", 0, 0); err == nil {
		t.Error("zero rate accepted")
	}
}

func TestRecordsAreValidJSONWithTheDocumentedSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "r.jsonl.gz")
	w, err := NewWriter(path, "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	w.Add(Record{TS: 1793667361.956472, Row: 42, Op: "read", Outcome: Outcome{Outcome: OutcomeOK}, LatencyMS: 1.25, Weight: 1})
	w.Add(Record{TS: 1.5, Row: 7, Op: "write", Outcome: Classify(&mysql.MySQLError{Number: 9005, Message: "Region is \"unavailable\"\nretry"}, false),
		LatencyMS: 5000, Err: "Error 9005: Region is \"unavailable\"\nretry", Weight: 1})
	w.Add(Record{TS: 2, Row: 1, Op: "read", Outcome: Outcome{Outcome: OutcomeOK}, LatencyMS: 2, Weight: 10})
	if n := w.Close(); n != 3 {
		t.Fatalf("wrote %d records", n)
	}
	f, _ := os.Open(path)
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var recs []map[string]any
	sc := bufio.NewScanner(gz)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("invalid JSON %q: %v", sc.Text(), err)
		}
		recs = append(recs, m)
	}
	if len(recs) != 3 {
		t.Fatalf("read %d records", len(recs))
	}
	ok, bad, weighted := recs[0], recs[1], recs[2]
	if ok["outcome"] != "ok" || ok["client_region"] != "us-east-1" || ok["row"].(float64) != 42 || ok["cause"] != nil || ok["err"] != nil {
		t.Errorf("ok record: %v", ok)
	}
	if bad["outcome"] != "timeout" || bad["cause"] != "tidb_9005_region_unavailable" || bad["code"].(float64) != 9005 ||
		bad["layer"] != "tikv" || !strings.Contains(bad["err"].(string), "unavailable") || bad["w"] != nil {
		t.Errorf("failure record: %v", bad)
	}
	if weighted["w"].(float64) != 10 {
		t.Errorf("weight: %v", weighted)
	}
	if ok["ts"].(float64) != 1793667361.956472 {
		t.Errorf("timestamp precision: %v", ok["ts"])
	}
}

func TestTruncateCollapsesWhitespaceAndLimitsLength(t *testing.T) {
	if got := truncate("a  b\n c", 200); got != "a b c" {
		t.Errorf("got %q", got)
	}
	if got := truncate(strings.Repeat("x", 500), 200); len([]rune(got)) != 200 {
		t.Errorf("len %d", len([]rune(got)))
	}
}
