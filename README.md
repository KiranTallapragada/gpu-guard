# gpu-guard

**Is your self-hosted LLM really running on the GPU?**

A model server can lose its GPU (after a driver update, a suspend and resume, or a container
runtime restart) and carry on serving from the CPU. The API still answers, so a normal health
check stays green, but every request is many times slower.

`gpu-guard` checks what matters: it sends a tiny fixed prompt and measures **tokens per second**
against a threshold from your own healthy server. With Ollama it can also read how much of each
loaded model is in GPU memory. It works with vLLM, Ollama or any OpenAI-compatible server, and
it's a single binary with no dependencies.

> **Status:** early (0.1, in development). Tested against simulated servers; real-GPU testing on
> vLLM and Ollama is next. Recovery actions are planned, not built yet.

## Use it

```
# 1. While the model is known to be on the GPU, measure it:
gpu-guard calibrate --url http://localhost:8000/v1

run 1: 61.8 tokens/s
...
Suggested: gpu-guard check --url http://localhost:8000/v1 --model Qwen/Qwen3-8B --min-tps 30.4

# 2. Check it any time. Exit code 0 = healthy, 1 = unhealthy:
gpu-guard check --url http://localhost:8000/v1 --min-tps 30
```

The numbers above are illustrative. The threshold is half your median healthy speed: normal
variation stays above it, and a CPU fallback falls far below.

**Ollama:** add the placement check. After the probe, it fails if the probed model isn't loaded or
any loaded model isn't fully in GPU memory:

```
gpu-guard check --url http://localhost:11434/v1 --ollama http://localhost:11434 --min-tps 20
```

Add `--json` to `check` for machine-readable output, and `--api-key` (or `GPU_GUARD_API_KEY`) if
your server needs one. The key is never printed, and redirects are not followed, so it isn't sent
anywhere except the URL you give.

## As a health check

**Docker Compose**

```yaml
healthcheck:
  test: ["CMD", "gpu-guard", "check", "--url", "http://localhost:8000/v1", "--min-tps", "30"]
  interval: 2m
  timeout: 90s
  retries: 3
```

**Kubernetes**

```yaml
livenessProbe:
  exec:
    command: ["gpu-guard", "check", "--url", "http://localhost:8000/v1", "--min-tps", "30"]
  periodSeconds: 120
  timeoutSeconds: 90
  failureThreshold: 3
```

Each check generates a few dozen tokens, so keep the interval in minutes, not seconds.

## Install

```
go install github.com/KiranTallapragada/gpu-guard@latest
```

## What it checks, and what it doesn't

- **Speed:** tokens per second for a short, fixed prompt (including prompt processing, which is
  small here). Busy servers are slower too, so set the threshold with some headroom and use a
  few retries before acting.
- **Ollama placement:** `size_vram` against `size` from `/api/ps`.
- It does not inspect drivers or CUDA directly; it measures the symptom users feel.

## Roadmap

- [x] Speed probe, calibration, Ollama placement check, JSON output
- [ ] Recovery: run your own commands in order (restart the server, reload the driver module,
      then alert), with cooldowns and a limit on attempts
- [ ] vLLM metrics (KV-cache usage, waiting requests) to tell "busy" apart from "degraded"
- [ ] Release binaries for Linux (amd64, arm64)

## Licence

MIT
