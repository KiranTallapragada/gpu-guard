// gpu-guard: fail a health check when a self-hosted LLM has quietly fallen back to the CPU.
//
// Works with vLLM, Ollama or any OpenAI-compatible server. One static binary, no dependencies.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"time"
)

var version = "0.1.0-dev"

const usage = `gpu-guard: is the model really running on the GPU?

Usage:
  gpu-guard check     [flags]   one check; exit 0 healthy, 1 unhealthy (for Docker/Kubernetes)
  gpu-guard calibrate [flags]   measure a healthy server and suggest --min-tps
  gpu-guard version

Run "gpu-guard <command> -h" for the flags.
`

type common struct {
	url, model, apiKey string
	maxTokens          int
	timeout            time.Duration
	jsonOut            bool
}

func addCommon(fs *flag.FlagSet, c *common) {
	fs.StringVar(&c.url, "url", "http://localhost:8000/v1", "OpenAI-compatible base URL (Ollama: http://localhost:11434/v1)")
	fs.StringVar(&c.model, "model", "", "model name (default: first model the server lists)")
	fs.StringVar(&c.apiKey, "api-key", "", "API key (default: the GPU_GUARD_API_KEY environment variable)")
	fs.IntVar(&c.maxTokens, "max-tokens", 64, "tokens to generate per probe")
	fs.DurationVar(&c.timeout, "timeout", 60*time.Second, "time limit per probe")
	fs.BoolVar(&c.jsonOut, "json", false, "machine-readable output")
}

func (c common) probe() Probe {
	key := c.apiKey
	if key == "" {
		key = os.Getenv("GPU_GUARD_API_KEY") // read after parsing, so it never appears in help output
	}
	return Probe{BaseURL: c.url, Model: c.model, APIKey: key, MaxTokens: c.maxTokens, Timeout: c.timeout}
}

// validate rejects settings that would make a check meaningless.
func (c common) validate(fs *flag.FlagSet) error {
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected argument %q (flags go before nothing else)", fs.Arg(0))
	}
	if c.maxTokens < 8 {
		return fmt.Errorf("--max-tokens must be at least 8")
	}
	if c.timeout <= 0 {
		return fmt.Errorf("--timeout must be positive")
	}
	return nil
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// parseArgs returns (exit code, done). Help is a success, other parse errors are usage errors.
func parseArgs(fs *flag.FlagSet, args []string) (int, bool) {
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, true
		}
		return 2, true
	}
	return 0, false
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	switch args[0] {
	case "check":
		return cmdCheck(args[1:], stdout, stderr)
	case "calibrate":
		return cmdCalibrate(args[1:], stdout, stderr)
	case "version", "--version", "-v":
		fmt.Fprintln(stdout, "gpu-guard", version)
		return 0
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usage)
		return 0
	}
	fmt.Fprintf(stderr, "unknown command %q\n\n%s", args[0], usage)
	return 2
}

func cmdCheck(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c common
	addCommon(fs, &c)
	minTPS := fs.Float64("min-tps", 0, "fail below this many tokens/s (get a value from 'gpu-guard calibrate')")
	ollama := fs.String("ollama", "", "also check Ollama's GPU placement, e.g. http://localhost:11434")
	minShare := fs.Float64("min-gpu-share", 1.0, "Ollama: share of each model that must be in GPU memory (0-1)")
	if code, done := parseArgs(fs, args); done {
		return code
	}
	if err := c.validate(fs); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if !finite(*minTPS) || *minTPS < 0 || !finite(*minShare) || *minShare < 0 || *minShare > 1 {
		fmt.Fprintln(stderr, "--min-tps must be a number >= 0 and --min-gpu-share between 0 and 1")
		return 2
	}
	if *minTPS <= 0 && *ollama == "" {
		fmt.Fprintln(stderr, "set --min-tps (run 'gpu-guard calibrate' first) and/or --ollama")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout+10*time.Second)
	defer cancel()
	res := Check(ctx, CheckConfig{Probe: c.probe(), MinTPS: *minTPS, MinTokens: 8, OllamaURL: *ollama, MinGPUShare: *minShare})

	if c.jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		if res.Probe != nil {
			fmt.Fprintf(stdout, "%s: %d tokens in %.2fs = %.1f tokens/s\n", res.Probe.Model, res.Probe.CompletionTokens, res.Probe.Seconds, res.Probe.TokensPerSecond)
		}
		for _, m := range res.Ollama {
			fmt.Fprintf(stdout, "ollama %s: %.0f%% in GPU memory\n", m.Name, m.GPUShare()*100)
		}
		if res.Status == Healthy {
			fmt.Fprintln(stdout, "✓ healthy")
		} else {
			for _, r := range res.Reasons {
				fmt.Fprintln(stdout, "✗", r)
			}
		}
	}
	if res.Status != Healthy {
		return 1
	}
	return 0
}

func cmdCalibrate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("calibrate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c common
	addCommon(fs, &c)
	runs := fs.Int("runs", 5, "number of probes")
	if code, done := parseArgs(fs, args); done {
		return code
	}
	if err := c.validate(fs); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if *runs < 2 {
		fmt.Fprintln(stderr, "--runs must be at least 2 (the first run is a warm-up)")
		return 2
	}
	p := c.probe()
	var speeds []float64
	for i := 0; i < *runs; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), c.timeout+10*time.Second)
		r, err := p.Run(ctx)
		cancel()
		if err != nil {
			fmt.Fprintln(stderr, "probe failed:", err)
			return 1
		}
		p.Model = r.Model
		if r.CompletionTokens < 8 || r.TokensPerSecond <= 0 {
			fmt.Fprintf(stderr, "run %d produced %d tokens; too few to measure (raise --max-tokens)\n", i, r.CompletionTokens)
			return 1
		}
		if i == 0 {
			continue // first request may include warm-up
		}
		speeds = append(speeds, r.TokensPerSecond)
		fmt.Fprintf(stdout, "run %d: %.1f tokens/s\n", i, r.TokensPerSecond)
	}
	s := SuggestMinTPS(speeds)
	fmt.Fprintf(stdout, "\nSuggested: gpu-guard check --url %s --model %s --min-tps %.1f\n", c.url, p.Model, s)
	fmt.Fprintln(stdout, "(half the median healthy speed; run this while the model is known to be on the GPU)")
	return 0
}
