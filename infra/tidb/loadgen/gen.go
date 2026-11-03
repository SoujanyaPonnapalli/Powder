package main

// Key distributions and arrival-rate profiles: Goal 6.3's four workloads (uniform, diurnal, hot partition,
// shifting hotspot) are configurations of these two pieces.

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type KeyGen interface {
	// Next returns a row id in [0, keys) for a request issued at offset t after the start of the run.
	Next(rng *rand.Rand, t time.Duration) int64
}

type uniformKeys struct{ keys int64 }

func (u uniformKeys) Next(rng *rand.Rand, _ time.Duration) int64 { return rng.Int63n(u.keys) }

// zipfKeys concentrates traffic on low row ids: with a bucketed key space that is a hot partition.
type zipfKeys struct {
	keys int64
	s, v float64
}

func (z zipfKeys) Next(rng *rand.Rand, _ time.Duration) int64 {
	// rand.Zipf is stateful per rng; build on demand (cheap) so each worker's rng stays independent.
	zf := rand.NewZipf(rng, z.s, z.v, uint64(z.keys-1))
	return int64(zf.Uint64())
}

// hotRange sends hotFraction of requests to a window of width*keys rows at a fixed or shifting offset.
type hotRange struct {
	keys     int64
	fraction float64
	width    int64
	shift    time.Duration // 0 = fixed window at the start of the key space
}

func (h hotRange) Next(rng *rand.Rand, t time.Duration) int64 {
	if rng.Float64() >= h.fraction {
		return rng.Int63n(h.keys)
	}
	var off int64
	if h.shift > 0 {
		step := int64(t / h.shift)
		off = (step * h.width) % h.keys
	}
	return (off + rng.Int63n(h.width)) % h.keys
}

func NewKeyGen(kind string, keys int64, hotFraction, hotWidth float64, shift time.Duration) (KeyGen, error) {
	if keys < 2 {
		return nil, fmt.Errorf("keys must be >= 2")
	}
	w := int64(math.Max(1, hotWidth*float64(keys)))
	switch kind {
	case "uniform":
		return uniformKeys{keys}, nil
	case "zipf":
		return zipfKeys{keys, 1.1, 1}, nil
	case "hot-range":
		return hotRange{keys, hotFraction, w, 0}, nil
	case "shifting-hot-range":
		if shift <= 0 {
			return nil, fmt.Errorf("shifting-hot-range needs --shift-every > 0")
		}
		return hotRange{keys, hotFraction, w, shift}, nil
	}
	return nil, fmt.Errorf("unknown key distribution %q", kind)
}

// Rate returns the arrival rate (requests/s) at offset t.
type Rate func(t time.Duration) float64

func NewRate(kind string, base float64, period time.Duration) (Rate, error) {
	if base <= 0 {
		return nil, fmt.Errorf("rate must be positive")
	}
	switch kind {
	case "constant":
		return func(time.Duration) float64 { return base }, nil
	case "diurnal":
		if period <= 0 {
			return nil, fmt.Errorf("diurnal needs --period > 0")
		}
		// base * (1 + 0.6 sin): swings between 0.4x and 1.6x, mean = base
		return func(t time.Duration) float64 {
			return base * (1 + 0.6*math.Sin(2*math.Pi*float64(t)/float64(period)))
		}, nil
	}
	return nil, fmt.Errorf("unknown arrival profile %q", kind)
}
