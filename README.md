# gpu-guard

**Is your self-hosted LLM really running on the GPU?**

A model server can lose its GPU (after a driver update, a suspend and resume, or a container
runtime restart) and carry on serving from the CPU. The API still answers, so a normal health
check stays green, but every request is many times slower.

`gpu-guard` checks what matters: it sends a tiny fixed prompt and measures **tokens per second**
against a threshold from your own healthy server. With Ollama it can also read how much of each
loaded model is in GPU memory. It's designed for vLLM, Ollama and other servers with
OpenAI-compatible `/models` and `/chat/completions` endpoints (real-server testing is pending), and
it's a single binary with no dependencies.

> **Status:** early (0.1, in development). Tested against simulated servers; real-GPU testing on
> vLLM and Ollama is next. Recovery (`watch`) is new and tested the same way.

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

The numbers above are illustrative. Half your median healthy speed is a starting threshold: check
it under normal load, and lower it if busy periods trip it. A model running on the CPU is usually
many times slower, but measure your own case.

**Ollama:** add the placement check. After the probe, it fails if the probed model isn't loaded or
any loaded model isn't fully in GPU memory:

```
gpu-guard check --url http://localhost:11434/v1 --ollama http://localhost:11434 --min-tps 20
```

Add `--json` to `check` for machine-readable output, and `--api-key` (or `GPU_GUARD_API_KEY`) if
your server needs one. The key is not printed (it's redacted if a server echoes it in an error),
and redirects are not followed, so it isn't sent anywhere except the URL you give. Pass
`--api-key=` to send no key even when the environment variable is set.

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

## Recover automatically (`watch`)

`gpu-guard watch` runs the same check on a schedule. After several failures in a row it runs
**your** recovery commands, one step per escalation, mildest first:

```
gpu-guard watch --url http://localhost:8000/v1 --min-tps 30 \
  --on-fail "docker restart vllm" \
  --on-fail "sudo /usr/local/sbin/reload-nvidia-uvm" \
  --on-fail "/usr/local/bin/page-oncall 'GPU still degraded'"
```

- **Escalation:** after `--failures` (default 3) failed checks it runs step 1; if the server is still
  unhealthy after `--settle`, step 2, and so on. The last step repeats if needed. One healthy check
  resets everything to step 1.
- **Safety limits:** at most one step per `--cooldown` (default 10 minutes) and `--max-per-hour`
  (default 3), so a persistent fault can't cause a storm of restarts. Each command is stopped after
  `--step-timeout` (default 5 minutes).
- **Try it first:** `--dry-run` logs what would run without running anything.
- **Log:** one JSON line per event (`healthy`, `unhealthy`, `action`, `action_failed`, `held`,
  `dry_run`) on standard output, ready for any log collector.

gpu-guard runs nothing you didn't configure. Commands run with `/bin/sh -c` as the user running
gpu-guard, so give that user only the permissions the commands need (for example a narrow sudoers
rule for one script rather than full root).

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
- [x] Recovery: your own commands in order, with cooldowns, an hourly limit and a dry run (`watch`)
- [ ] vLLM metrics (KV-cache usage, waiting requests) to tell "busy" apart from "degraded"
- [ ] Release binaries for Linux (amd64, arm64)

## Licence

MIT
