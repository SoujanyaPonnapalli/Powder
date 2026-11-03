package main

// Classification of a finished request into ok|timeout|conflict|other with a recorded cause.
// The code table is error_classes.json, a byte-for-byte copy of powder/deploy/error_classes.json (a Python test
// fails if they drift), so the Go generator and the Python analysis can never disagree about what counts as a
// timeout. Precedence: client deadline, then known MySQL/TiDB code, then wording, then "other" carrying the code.

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"regexp"
	"strconv"

	"github.com/go-sql-driver/mysql"
)

//go:embed error_classes.json
var classTableJSON []byte

type classEntry struct {
	Outcome string `json:"outcome"`
	Cause   string `json:"cause"`
	Layer   string `json:"layer"`
}

type textPattern struct {
	Regex string `json:"regex"`
	classEntry
	re *regexp.Regexp
}

type classTable struct {
	MySQLCodes     map[string]classEntry `json:"mysql_codes"`
	ClientDeadline classEntry            `json:"client_deadline"`
	Patterns       []textPattern         `json:"error_text_patterns"`
	UnclassPrefix  string                `json:"unclassified_cause_prefix"`
}

var table = mustLoadTable()

func mustLoadTable() *classTable {
	var t classTable
	if err := json.Unmarshal(classTableJSON, &t); err != nil {
		panic("error_classes.json: " + err.Error())
	}
	for i := range t.Patterns {
		t.Patterns[i].re = regexp.MustCompile("(?i)" + t.Patterns[i].Regex)
	}
	return &t
}

// Outcome is the classified result of one request. Code is 0 when there is none.
type Outcome struct {
	Outcome string
	Cause   string
	Code    int
	Layer   string
}

const (
	OutcomeOK       = "ok"
	OutcomeTimeout  = "timeout"
	OutcomeConflict = "conflict"
	OutcomeOther    = "other"
)

// Classify classifies err (nil = success). deadline reports that the client's own per-request deadline expired.
func Classify(err error, deadline bool) Outcome {
	if err == nil {
		return Outcome{Outcome: OutcomeOK}
	}
	code := 0
	var me *mysql.MySQLError
	if errors.As(err, &me) {
		code = int(me.Number)
	}
	return classifyRaw(code, err.Error(), deadline || errors.Is(err, context.DeadlineExceeded))
}

func classifyRaw(code int, text string, deadline bool) Outcome {
	if deadline {
		d := table.ClientDeadline
		return Outcome{d.Outcome, d.Cause, code, d.Layer}
	}
	if code != 0 {
		if e, ok := table.MySQLCodes[strconv.Itoa(code)]; ok {
			return Outcome{e.Outcome, e.Cause, code, e.Layer}
		}
	}
	if text != "" {
		for _, p := range table.Patterns {
			if p.re.MatchString(text) {
				return Outcome{p.Outcome, p.Cause, code, p.Layer}
			}
		}
	}
	suffix := "no_code"
	if code != 0 {
		suffix = strconv.Itoa(code)
	}
	return Outcome{OutcomeOther, table.UnclassPrefix + "_" + suffix, code, ""}
}
