package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

// newTestWatcher returns a watcher whose checks follow `health` (true = healthy) and whose
// steps are recorded instead of executed.
func newTestWatcher(health *[]bool, ran *[]string) (*Watcher, *fakeClock, *bytes.Buffer) {
	clk := &fakeClock{t: time.Date(2026, 11, 20, 12, 0, 0, 0, time.UTC)}
	log := &bytes.Buffer{}
	w := &Watcher{
		Check: func(context.Context) CheckResult {
			ok := (*health)[0]
			*health = (*health)[1:]
			if ok {
				return CheckResult{Status: Healthy}
			}
			return CheckResult{Status: Unhealthy, Reasons: []string{"slow"}}
		},
		Steps:    []string{"restart", "reload-driver", "alert"},
		Failures: 2, Interval: time.Minute, Settle: 3 * time.Minute,
		Cooldown: 10 * time.Minute, MaxPerHour: 3, StepTimeout: time.Minute,
		Log: log, Now: clk.now,
		Sleep: func(context.Context, time.Duration) bool { return true },
		RunStep: func(_ context.Context, cmd string) (string, error) {
			*ran = append(*ran, cmd)
			return "ok", nil
		},
	}
	return w, clk, log
}

func events(t *testing.T, log *bytes.Buffer) []Event {
	t.Helper()
	var out []Event
	for _, line := range strings.Split(strings.TrimSpace(log.String()), "\n") {
		var e Event
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("bad log line %q: %v", line, err)
		}
		out = append(out, e)
	}
	return out
}

func TestNoActionBeforeEnoughFailures(t *testing.T) {
	health := []bool{false}
	var ran []string
	w, _, _ := newTestWatcher(&health, &ran)
	w.Tick(context.Background())
	if len(ran) != 0 {
		t.Fatalf("acted after one failure: %v", ran)
	}
}

func TestEscalatesInOrderWithCooldown(t *testing.T) {
	health := []bool{false, false, false, false, false, false, false, false}
	var ran []string
	w, clk, _ := newTestWatcher(&health, &ran)
	ctx := context.Background()
	for i := 0; i < 2; i++ { // two failures -> step 1
		w.Tick(ctx)
		clk.advance(time.Minute)
	}
	w.Tick(ctx) // failure 1 of the next round
	clk.advance(time.Minute)
	w.Tick(ctx) // failure 2, but cooldown (10 min) not over: held
	if len(ran) != 1 || ran[0] != "restart" {
		t.Fatalf("after cooldown hold: %v", ran)
	}
	clk.advance(10 * time.Minute)
	w.Tick(ctx) // still failing, cooldown over -> step 2
	if len(ran) != 2 || ran[1] != "reload-driver" {
		t.Fatalf("escalation: %v", ran)
	}
}

func TestHealthyResetsEscalation(t *testing.T) {
	health := []bool{false, false, true, false, false}
	var ran []string
	w, clk, log := newTestWatcher(&health, &ran)
	ctx := context.Background()
	for range 5 {
		w.Tick(ctx)
		clk.advance(30 * time.Minute)
	}
	if len(ran) != 2 || ran[0] != "restart" || ran[1] != "restart" {
		t.Fatalf("want restart twice after recovery reset, got %v", ran)
	}
	if !strings.Contains(log.String(), "escalation reset") {
		t.Fatalf("missing reset note: %s", log.String())
	}
}

func TestHourlyCapAndLastStepRepeats(t *testing.T) {
	health := make([]bool, 40)
	var ran []string
	w, clk, log := newTestWatcher(&health, &ran)
	w.Cooldown = 0
	w.Failures = 1
	ctx := context.Background()
	for range 6 { // six failures within the same hour
		w.Tick(ctx)
		clk.advance(time.Minute)
	}
	if len(ran) != 3 {
		t.Fatalf("hourly cap of 3 not enforced: %v", ran)
	}
	if !strings.Contains(log.String(), "hourly limit reached") {
		t.Fatalf("missing hold reason")
	}
	clk.advance(time.Hour)
	before := len(ran)
	w.Tick(ctx)
	if len(ran) != before+1 || ran[len(ran)-1] != "alert" {
		t.Fatalf("after the last step a new run of the strongest step is expected, got %v", ran)
	}
}

func TestDryRunRunsNothing(t *testing.T) {
	health := []bool{false, false}
	var ran []string
	w, _, log := newTestWatcher(&health, &ran)
	w.DryRun = true
	w.Tick(context.Background())
	w.Tick(context.Background())
	if len(ran) != 0 {
		t.Fatalf("dry run executed %v", ran)
	}
	ev := events(t, log)
	if last := ev[len(ev)-1]; last.Event != "dry_run" || last.Command != "restart" || last.Step != 1 {
		t.Fatalf("got %+v", last)
	}
}

func TestFailedStepIsLogged(t *testing.T) {
	health := []bool{false, false}
	var ran []string
	w, _, log := newTestWatcher(&health, &ran)
	w.RunStep = func(context.Context, string) (string, error) { return "boom", errors.New("exit status 1") }
	w.Tick(context.Background())
	w.Tick(context.Background())
	ev := events(t, log)
	if last := ev[len(ev)-1]; last.Event != "action_failed" || last.Note != "exit status 1" {
		t.Fatalf("got %+v", last)
	}
}

func TestShellStepRunsTheCommand(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "ran")
	if _, err := shellStep(context.Background(), "echo hi > "+marker); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(marker); strings.TrimSpace(string(b)) != "hi" {
		t.Fatalf("command did not run")
	}
}

func TestShellStepTimesOut(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := shellStep(ctx, "sleep 5"); err == nil || time.Since(start) > 3*time.Second {
		t.Fatalf("step should be killed by the timeout (err=%v, took %v)", err, time.Since(start))
	}
}

func TestWatchUsageErrors(t *testing.T) {
	for _, args := range [][]string{
		{"watch", "--min-tps", "10", "--failures", "0"},
		{"watch", "--min-tps", "10", "--interval", "0s"},
		{"watch", "--min-tps", "10", "--on-fail", " "},
		{"watch"},
	} {
		var out, errb bytes.Buffer
		if code := run(args, &out, &errb); code != 2 {
			t.Errorf("%v: exit %d, want 2", args, code)
		}
	}
}

// End to end: real Check against a slow server, real watcher loop, real shell commands.
func TestWatchEndToEndRunsRealStepsInOrder(t *testing.T) {
	s := fakeServer(t, 16, 200*time.Millisecond, "") // ~80 tokens/s, below the 1000 minimum
	marker := filepath.Join(t.TempDir(), "marker")
	cfg := cfg(s.URL, 1000)
	log := &bytes.Buffer{}
	w := &Watcher{
		Check:    func(ctx context.Context) CheckResult { return Check(ctx, cfg) },
		Steps:    []string{"echo step1 >> " + marker, "echo step2 >> " + marker},
		Failures: 2, Interval: 10 * time.Millisecond, Settle: 10 * time.Millisecond,
		MaxPerHour: 3, StepTimeout: 5 * time.Second,
		Log: log, Now: time.Now, Sleep: sleepCtx, RunStep: shellStep,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second) // watchdog only
	defer cancel()
	w.Sleep = func(ctx context.Context, d time.Duration) bool {
		if strings.Contains(log.String(), "hourly limit reached") {
			cancel() // done: three actions ran and the cap held the next one
		}
		return sleepCtx(ctx, d)
	}
	w.Run(ctx)
	b, _ := os.ReadFile(marker)
	got := strings.Fields(string(b))
	if len(got) != 3 || got[0] != "step1" || got[1] != "step2" || got[2] != "step2" {
		t.Fatalf("want step1, step2, step2 (capped at 3/hour), got %v\nlog:\n%s", got, log.String())
	}
	if !strings.Contains(log.String(), "hourly limit reached") {
		t.Fatalf("expected the hourly cap to hold further steps")
	}
}

func TestCooldownLongerThanAnHourHolds(t *testing.T) {
	health := make([]bool, 10)
	var ran []string
	w, clk, log := newTestWatcher(&health, &ran)
	w.Failures, w.Cooldown, w.MaxPerHour = 1, 2*time.Hour, 0
	ctx := context.Background()
	w.Tick(ctx)
	clk.advance(61 * time.Minute)
	w.Tick(ctx)
	if len(ran) != 1 || !strings.Contains(log.String(), `"note":"cooldown"`) {
		t.Fatalf("2h cooldown must still hold after 61 minutes: ran %v", ran)
	}
	clk.advance(60 * time.Minute)
	w.Tick(ctx)
	if len(ran) != 2 {
		t.Fatalf("cooldown over, a second step was expected: %v", ran)
	}
}

func TestCancelledCheckIsNotAFailure(t *testing.T) {
	health := []bool{false, false}
	var ran []string
	w, _, log := newTestWatcher(&health, &ran)
	w.Failures = 1
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w.Tick(ctx)
	if len(ran) != 0 || log.Len() != 0 {
		t.Fatalf("a cancelled tick must log and run nothing: ran %v, log %q", ran, log.String())
	}
}

func TestRunWaitsIntervalOrSettle(t *testing.T) {
	health := []bool{true, false, false, false}
	var ran []string
	w, _, _ := newTestWatcher(&health, &ran)
	var waits []time.Duration
	w.Sleep = func(_ context.Context, d time.Duration) bool {
		waits = append(waits, d)
		return len(health) > 0
	}
	w.Run(context.Background())
	// healthy -> interval; failure 1 -> interval; failure 2 -> action -> settle; failure -> interval
	want := []time.Duration{time.Minute, time.Minute, 3 * time.Minute, time.Minute}
	if len(waits) != len(want) {
		t.Fatalf("waits %v, want %v", waits, want)
	}
	for i := range want {
		if waits[i] != want[i] {
			t.Fatalf("wait %d = %v, want %v (all: %v)", i, waits[i], want[i], waits)
		}
	}
}

func TestStepOutputIsCapped(t *testing.T) {
	out, err := shellStep(context.Background(), "head -c 1000000 /dev/zero | tr '\\0' x")
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > maxStepOutput+64 || !strings.HasSuffix(out, "[output truncated]") {
		t.Fatalf("output not capped: %d bytes", len(out))
	}
}
