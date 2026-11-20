package main

import (
	"context"
	"fmt"
	"net/http"
	"sort"
)

type Status string

const (
	Healthy   Status = "healthy"
	Unhealthy Status = "unhealthy"
)

type CheckConfig struct {
	Probe        Probe
	MinTPS       float64 // fail below this many tokens per second
	MinTokens    int     // fewer completion tokens than this: too short to judge speed
	OllamaURL    string  // optional, e.g. http://localhost:11434
	MinGPUShare  float64 // Ollama: fail when less than this share of a model is in GPU memory
	OllamaClient *http.Client
}

type CheckResult struct {
	Status  Status        `json:"status"`
	Reasons []string      `json:"reasons,omitempty"`
	Probe   *ProbeResult  `json:"probe,omitempty"`
	Ollama  []OllamaModel `json:"ollama,omitempty"`
}

func (r *CheckResult) fail(format string, a ...any) {
	r.Status = Unhealthy
	r.Reasons = append(r.Reasons, fmt.Sprintf(format, a...))
}

// Check runs the speed probe and, when configured, Ollama's GPU-placement check.
func Check(ctx context.Context, c CheckConfig) CheckResult {
	res := CheckResult{Status: Healthy}

	if c.OllamaURL != "" {
		client := c.OllamaClient
		if client == nil {
			client = c.Probe.client()
		}
		models, err := OllamaLoaded(ctx, client, c.OllamaURL)
		if err != nil {
			res.fail("could not read Ollama model placement: %v", err)
		} else {
			res.Ollama = models
			for _, m := range models {
				if share := m.GPUShare(); share < c.MinGPUShare {
					res.fail("%s: only %.0f%% of the model is in GPU memory (CPU fallback)", m.Name, share*100)
				}
			}
		}
	}

	pr, err := c.Probe.Run(ctx)
	if err != nil {
		res.fail("probe request failed: %v", err)
		return res
	}
	res.Probe = &pr
	switch {
	case pr.CompletionTokens < c.MinTokens:
		res.fail("only %d tokens generated; too few to measure speed (raise --max-tokens or check the model)", pr.CompletionTokens)
	case c.MinTPS > 0 && pr.TokensPerSecond < c.MinTPS:
		res.fail("%.1f tokens/s is below the minimum of %.1f (possible CPU fallback or a degraded GPU)", pr.TokensPerSecond, c.MinTPS)
	}
	return res
}

// SuggestMinTPS turns a few healthy measurements into a threshold: half the median speed.
// Normal variation stays well above it; a CPU fallback (typically 10x slower or worse) falls far below.
func SuggestMinTPS(samples []float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	s := append([]float64(nil), samples...)
	sort.Float64s(s)
	mid := s[len(s)/2]
	if len(s)%2 == 0 {
		mid = (s[len(s)/2-1] + s[len(s)/2]) / 2
	}
	return mid / 2
}
