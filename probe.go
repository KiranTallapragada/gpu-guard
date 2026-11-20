package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Probe asks an OpenAI-compatible server for a short, fixed completion and measures how fast
// tokens come back. A model that has fallen back to the CPU still answers, just many times
// slower, so speed is the signal, not liveness.
type Probe struct {
	BaseURL   string // e.g. http://localhost:8000/v1
	Model     string // empty: first model from GET /models
	APIKey    string
	MaxTokens int
	Timeout   time.Duration
	Client    *http.Client
}

type ProbeResult struct {
	Model            string  `json:"model"`
	CompletionTokens int     `json:"completion_tokens"`
	Seconds          float64 `json:"seconds"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
}

const probePrompt = "Count upwards from one, separating the numbers with commas, and do not stop early."

func (p *Probe) client() *http.Client {
	if p.Client != nil {
		return p.Client
	}
	return &http.Client{Timeout: p.Timeout}
}

func (p *Probe) do(ctx context.Context, method, path string, body any, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(p.BaseURL, "/")+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	resp, err := p.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		return fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (p *Probe) firstModel(ctx context.Context) (string, error) {
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := p.do(ctx, http.MethodGet, "/models", nil, &models); err != nil {
		return "", err
	}
	if len(models.Data) == 0 {
		return "", fmt.Errorf("the server lists no models")
	}
	return models.Data[0].ID, nil
}

// Run sends one probe request and returns the measured speed.
func (p *Probe) Run(ctx context.Context) (ProbeResult, error) {
	model := p.Model
	if model == "" {
		m, err := p.firstModel(ctx)
		if err != nil {
			return ProbeResult{}, fmt.Errorf("finding the model: %w", err)
		}
		model = m
	}
	req := map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": probePrompt}},
		"max_tokens":  p.MaxTokens,
		"temperature": 0,
		"stream":      false,
	}
	var resp struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	start := time.Now()
	if err := p.do(ctx, http.MethodPost, "/chat/completions", req, &resp); err != nil {
		return ProbeResult{Model: model}, err
	}
	secs := time.Since(start).Seconds()
	r := ProbeResult{Model: model, CompletionTokens: resp.Usage.CompletionTokens, Seconds: secs}
	if secs > 0 {
		r.TokensPerSecond = float64(r.CompletionTokens) / secs
	}
	return r, nil
}
