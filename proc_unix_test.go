//go:build unix

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// gone reports whether pid has exited. A zombie (exited, not yet reaped by its parent, e.g. a
// non-reaping container PID 1) counts as gone.
func gone(pid int) bool {
	out, err := exec.Command("ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
	st := strings.TrimSpace(string(out))
	return err != nil || st == "" || strings.HasPrefix(st, "Z")
}

func waitForPID(t *testing.T, file string) int {
	t.Helper()
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if b, err := os.ReadFile(file); err == nil && len(strings.TrimSpace(string(b))) > 0 {
			pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
			return pid
		}
	}
	t.Fatalf("process never wrote %s", file)
	return 0
}

func assertGone(t *testing.T, pid int) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if gone(pid) {
			return
		}
	}
	_ = exec.Command("kill", "-9", strconv.Itoa(pid)).Run()
	t.Fatalf("process %d survived", pid)
}

// A cancelled (timed-out or stopped) recovery command must not leave processes behind.
func TestCancelKillsChildAndGrandchild(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		_, _ = shellStep(ctx, "sh -c 'echo $$ > "+pidFile+"; exec sleep 30' & wait")
		close(done)
	}()
	pid := waitForPID(t, pidFile) // cancel only once the grandchild is running
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("shellStep did not return after cancellation")
	}
	assertGone(t, pid)
}

// A command that exits normally but leaves a background job must not leak it.
func TestBackgroundJobIsCleanedUpAfterNormalExit(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	cmd := "sh -c 'echo $$ > " + pidFile + "; exec sleep 30' >/dev/null 2>&1 & " +
		"while [ ! -s " + pidFile + " ]; do sleep 0.05; done" // exit only once the job is running
	if _, err := shellStep(context.Background(), cmd); err != nil {
		t.Fatal(err)
	}
	assertGone(t, waitForPID(t, pidFile))
}

// Cancelling after the command already finished is not reported as a failure.
func TestCancelAfterExitIsNotAnError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	out, err := shellStep(ctx, "echo done")
	cancel()
	if err != nil || strings.TrimSpace(out) != "done" {
		t.Fatalf("out %q err %v", out, err)
	}
}
