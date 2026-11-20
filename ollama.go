package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OllamaPlacement reads Ollama's /api/ps, which reports for each loaded model its total size
// and how much of it sits in GPU memory (size_vram). size_vram == 0 means the model is running
// entirely on the CPU; 0 < size_vram < size means it is split between CPU and GPU.
type OllamaModel struct {
	Name     string `json:"name"`
	Size     int64  `json:"size"`
	SizeVRAM int64  `json:"size_vram"`
}

func (m OllamaModel) GPUShare() float64 {
	if m.Size <= 0 {
		return 0
	}
	return float64(m.SizeVRAM) / float64(m.Size)
}

func OllamaLoaded(ctx context.Context, client *http.Client, baseURL string) ([]OllamaModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/api/ps", nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /api/ps: HTTP %d", resp.StatusCode)
	}
	var body struct {
		Models []OllamaModel `json:"models"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxBody)).Decode(&body); err != nil {
		return nil, err
	}
	return body.Models, nil
}
