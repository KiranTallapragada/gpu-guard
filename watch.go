package main

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"strings"
	"time"
)

// Watcher runs the health check on a schedule and, after enough failures in a row, runs the
// user's recovery commands in order, one step per escalation. It never runs anything the user
// didn't configure, and it is rate-limited so a persistent fault can't trigger a storm of restarts.
type Watcher struct {
	Check       func(context.Context) CheckResult
	Steps       []string      // recovery commands, mildest first (e.g. restart, reload driver, alert)
	Failures    int           // consecutive failed checks before acting
	Interval    time.Duration // between checks
	Settle      time.Duration // wait after an action before checking again
	Cooldown    time.Duration // minimum time between two actions
	MaxPerHour  int           // cap on actions in any rolling hour
	StepTimeout time.Duration // time limit for one recovery command
	DryRun      bool          // log what would run, run nothing

	Log     io.Writer
	Now     func() time.Time
	Sleep   func(context.Context, time.Duration) bool // false when ctx is done
	RunStep func(ctx context.Context, cmd string) (string, error)

	consecutive int
	nextStep    int
	actions     []time.Time // within the last hour, for the hourly cap
	lastAction  time.Time   // kept separately so cooldowns longer than an hour still hold
}

// Event is one JSON line in the watch log.
type Event struct {
	Time    string   `json:"time"`
	Event   string   `json:"event"` // healthy, unhealthy, action, action_failed, held, dry_run
	Reasons []string `json:"reasons,omitempty"`
	TPS     float64  `json:"tokens_per_second,omitempty"`
	Step    int      `json:"step,omitempty"`
	Command string   `json:"command,omitempty"`
	Output  string   `json:"output,omitempty"`
	Note    string   `json:"note,omitempty"`
}

func (w *Watcher) emit(e Event) {
	e.Time = w.Now().UTC().Format(time.RFC3339)
	b, _ := json.Marshal(e)
	_, _ = w.Log.Write(append(b, '\n'))
}

// canAct applies the cooldown and the hourly cap. It returns a reason when an action must wait.
func (w *Watcher) canAct(now time.Time) (bool, string) {
	recent := w.actions[:0]
	for _, t := range w.actions {
		if now.Sub(t) < time.Hour {
			recent = append(recent, t)
		}
	}
	w.actions = recent
	if !w.lastAction.IsZero() && now.Sub(w.lastAction) < w.Cooldown {
		return false, "cooldown"
	}
	if w.MaxPerHour > 0 && len(w.actions) >= w.MaxPerHour {
		return false, "hourly limit reached"
	}
	return true, ""
}

// Tick runs one check and, if warranted, one recovery step. It returns how long to wait before
// the next tick.
func (w *Watcher) Tick(ctx context.Context) time.Duration {
	res := w.Check(ctx)
	if ctx.Err() != nil {
		return 0 // stopping: a cancelled probe is not a failed check
	}
	ev := Event{Reasons: res.Reasons}
	if res.Probe != nil {
		ev.TPS = res.Probe.TokensPerSecond
	}
	if res.Status == Healthy {
		if w.consecutive > 0 || w.nextStep > 0 {
			ev.Note = "recovered; escalation reset"
		}
		w.consecutive, w.nextStep = 0, 0
		ev.Event = "healthy"
		w.emit(ev)
		return w.Interval
	}
	w.consecutive++
	ev.Event = "unhealthy"
	w.emit(ev)
	if w.consecutive < w.Failures || len(w.Steps) == 0 {
		return w.Interval
	}

	now := w.Now()
	if ok, why := w.canAct(now); !ok {
		w.emit(Event{Event: "held", Note: why})
		return w.Interval
	}
	i := w.nextStep
	if i >= len(w.Steps) {
		i = len(w.Steps) - 1 // stay on the last (strongest) step
	}
	cmd := w.Steps[i]
	w.actions = append(w.actions, now)
	w.lastAction = now
	w.consecutive = 0
	if w.nextStep < len(w.Steps) {
		w.nextStep++
	}
	if w.DryRun {
		w.emit(Event{Event: "dry_run", Step: i + 1, Command: cmd})
		return w.Settle
	}
	sctx, cancel := context.WithTimeout(ctx, w.StepTimeout)
	out, err := w.RunStep(sctx, cmd)
	cancel()
	e := Event{Event: "action", Step: i + 1, Command: cmd, Output: trim(out, 500)}
	if err != nil {
		e.Event, e.Note = "action_failed", err.Error()
	}
	w.emit(e)
	return w.Settle
}

// Run ticks until ctx is cancelled.
func (w *Watcher) Run(ctx context.Context) {
	for {
		wait := w.Tick(ctx)
		if !w.Sleep(ctx, wait) {
			return
		}
	}
}

// maxStepOutput caps how much of a recovery command's output is kept; the rest is discarded.
const maxStepOutput = 64 << 10

// shellStep runs one recovery command in its own process group, so a timeout or a stop
// signal ends the command and everything it started, not just the shell.
func shellStep(ctx context.Context, cmd string) (string, error) {
	c := exec.CommandContext(ctx, "/bin/sh", "-c", cmd)
	setProcessGroup(c)
	c.WaitDelay = 5 * time.Second // don't wait forever on pipes held open by stray processes
	var out capped
	c.Stdout, c.Stderr = &out, &out
	err := c.Run()
	cleanupGroup(c) // nothing it started outlives the step
	return out.String(), err
}

// capped keeps the first maxStepOutput bytes and silently drops the rest.
type capped struct {
	buf     []byte
	dropped bool
}

func (c *capped) Write(p []byte) (int, error) {
	if room := maxStepOutput - len(c.buf); room > 0 {
		if len(p) > room {
			c.buf, c.dropped = append(c.buf, p[:room]...), true
		} else {
			c.buf = append(c.buf, p...)
		}
	} else {
		c.dropped = true
	}
	return len(p), nil
}

func (c *capped) String() string {
	if c.dropped {
		return string(c.buf) + "\n[output truncated]"
	}
	return string(c.buf)
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func trim(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
