package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeServer imitates an OpenAI-compatible server that returns `tokens` completion tokens after `delay`.
func fakeServer(t *testing.T, tokens int, delay time.Duration, ollama string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"test-model"}]}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req["model"] != "test-model" {
			http.Error(w, "unknown model", http.StatusNotFound)
			return
		}
		time.Sleep(delay)
		_ = json.NewEncoder(w).Encode(map[string]any{"usage": map[string]int{"completion_tokens": tokens}})
	})
	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(ollama))
	})
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s
}

func cfg(url string, minTPS float64) CheckConfig {
	return CheckConfig{Probe: Probe{BaseURL: url + "/v1", MaxTokens: 64, Timeout: 5 * time.Second}, MinTPS: minTPS, MinTokens: 8, MinGPUShare: 1}
}

func TestFastServerIsHealthy(t *testing.T) {
	s := fakeServer(t, 64, 20*time.Millisecond, "")
	res := Check(context.Background(), cfg(s.URL, 100))
	if res.Status != Healthy || res.Probe.Model != "test-model" {
		t.Fatalf("got %+v", res)
	}
}

func TestSlowServerIsUnhealthy(t *testing.T) {
	s := fakeServer(t, 8, 400*time.Millisecond, "") // ~20 tokens/s
	res := Check(context.Background(), cfg(s.URL, 100))
	if res.Status != Unhealthy || !strings.Contains(res.Reasons[0], "below the minimum") {
		t.Fatalf("got %+v", res)
	}
}

func TestTooFewTokensIsUnhealthy(t *testing.T) {
	s := fakeServer(t, 2, 0, "")
	res := Check(context.Background(), cfg(s.URL, 1))
	if res.Status != Unhealthy || !strings.Contains(res.Reasons[0], "too few") {
		t.Fatalf("got %+v", res)
	}
}

func TestUnreachableServerIsUnhealthy(t *testing.T) {
	res := Check(context.Background(), cfg("http://127.0.0.1:1", 1))
	if res.Status != Unhealthy {
		t.Fatalf("got %+v", res)
	}
}

func TestOllamaCPUFallbackDetected(t *testing.T) {
	s := fakeServer(t, 64, 0, `{"models":[{"name":"test-model","size":6000000000,"size_vram":0}]}`)
	c := cfg(s.URL, 0)
	c.OllamaURL = s.URL
	res := Check(context.Background(), c)
	if res.Status != Unhealthy || !strings.Contains(res.Reasons[0], "0% of the model is in GPU memory") {
		t.Fatalf("got %+v", res)
	}
}

func TestOllamaFullyOnGPUIsHealthy(t *testing.T) {
	s := fakeServer(t, 64, 0, `{"models":[{"name":"test-model","size":6000000000,"size_vram":6000000000}]}`)
	c := cfg(s.URL, 0)
	c.OllamaURL = s.URL
	if res := Check(context.Background(), c); res.Status != Healthy {
		t.Fatalf("got %+v", res)
	}
}

func TestSuggestMinTPSIsHalfTheMedian(t *testing.T) {
	if got := SuggestMinTPS([]float64{40, 60, 50, 1000}); got != 27.5 {
		t.Fatalf("got %v", got)
	}
	if got := SuggestMinTPS([]float64{30, 50, 40}); got != 20 {
		t.Fatalf("got %v", got)
	}
}

func TestCheckCommandExitCodes(t *testing.T) {
	fast := fakeServer(t, 64, 10*time.Millisecond, "")
	var out, errb bytes.Buffer
	if code := run([]string{"check", "--url", fast.URL + "/v1", "--min-tps", "50"}, &out, &errb); code != 0 {
		t.Fatalf("healthy: exit %d, %s %s", code, out.String(), errb.String())
	}
	slow := fakeServer(t, 8, 400*time.Millisecond, "")
	out.Reset()
	if code := run([]string{"check", "--url", slow.URL + "/v1", "--min-tps", "100", "--json"}, &out, &errb); code != 1 {
		t.Fatalf("unhealthy: exit %d", code)
	}
	var res CheckResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil || res.Status != Unhealthy {
		t.Fatalf("json output: %v %s", err, out.String())
	}
	if code := run([]string{"check"}, &out, &errb); code != 2 {
		t.Fatalf("missing thresholds should be a usage error, got %d", code)
	}
}

func TestOllamaWrongOrEmptyModelListIsUnhealthy(t *testing.T) {
	for name, ps := range map[string]string{
		"other model loaded": `{"models":[{"name":"other:7b","size":1,"size_vram":1}]}`,
		"nothing loaded":     `{"models":[]}`,
	} {
		s := fakeServer(t, 64, 0, ps)
		c := cfg(s.URL, 0)
		c.OllamaURL = s.URL
		if res := Check(context.Background(), c); res.Status != Unhealthy {
			t.Errorf("%s: got %+v", name, res)
		}
	}
}

func TestHelpExitsZeroAndNeverShowsTheAPIKey(t *testing.T) {
	t.Setenv("GPU_GUARD_API_KEY", "sk-secret-value")
	for _, cmd := range []string{"check", "calibrate"} {
		var out, errb bytes.Buffer
		if code := run([]string{cmd, "-h"}, &out, &errb); code != 0 {
			t.Errorf("%s -h: exit %d", cmd, code)
		}
		if strings.Contains(out.String()+errb.String(), "sk-secret-value") {
			t.Errorf("%s -h printed the API key", cmd)
		}
	}
}

func TestAPIKeyFromEnvironmentIsSent(t *testing.T) {
	t.Setenv("GPU_GUARD_API_KEY", "sk-env")
	var got string
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"usage":{"completion_tokens":64}}`))
	}))
	defer s.Close()
	var out, errb bytes.Buffer
	run([]string{"check", "--url", s.URL + "/v1", "--model", "m", "--min-tps", "1"}, &out, &errb)
	if got != "Bearer sk-env" {
		t.Fatalf("Authorization = %q", got)
	}
}

func TestRedirectsAreNotFollowed(t *testing.T) {
	var leaked bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked = r.Header.Get("Authorization") != ""
	}))
	defer target.Close()
	redir := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redir.Close()
	c := cfg(redir.URL, 1)
	c.Probe.Model, c.Probe.APIKey = "m", "sk-x"
	if res := Check(context.Background(), c); res.Status != Unhealthy || leaked {
		t.Fatalf("status %v, key forwarded: %v", res.Status, leaked)
	}
}

func TestInvalidThresholdsAreUsageErrors(t *testing.T) {
	for _, args := range [][]string{
		{"check", "--min-tps", "NaN"},
		{"check", "--min-tps", "-5"},
		{"check", "--min-tps", "10", "--min-gpu-share", "2"},
		{"check", "--min-tps", "10", "--max-tokens", "2"},
		{"check", "--min-tps", "10", "stray"},
		{"calibrate", "--runs", "1"},
	} {
		var out, errb bytes.Buffer
		if code := run(args, &out, &errb); code != 2 {
			t.Errorf("%v: exit %d, want 2", args, code)
		}
	}
}

func TestCalibrateSuggestsAThreshold(t *testing.T) {
	s := fakeServer(t, 64, 20*time.Millisecond, "")
	var out, errb bytes.Buffer
	if code := run([]string{"calibrate", "--url", s.URL + "/v1", "--runs", "3"}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "--min-tps") || !strings.Contains(out.String(), "test-model") {
		t.Fatalf("output: %s", out.String())
	}
}

func TestCalibrateRejectsTooFewTokens(t *testing.T) {
	s := fakeServer(t, 2, 0, "")
	var out, errb bytes.Buffer
	if code := run([]string{"calibrate", "--url", s.URL + "/v1", "--runs", "2"}, &out, &errb); code != 1 {
		t.Fatalf("exit %d", code)
	}
}
