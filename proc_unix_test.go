//go:build unix

package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// A timed-out recovery command must not leave processes behind.
func TestTimeoutKillsChildAndGrandchild(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	cmd := "sh -c 'echo $$ > " + pidFile + "; exec sleep 30' & wait"
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, _ = shellStep(ctx, cmd)
	if time.Since(start) > 10*time.Second {
		t.Fatalf("shellStep did not return promptly after the timeout")
	}
	b, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("grandchild never started: %v", err)
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if err := syscall.Kill(pid, 0); err == syscall.ESRCH {
			return // gone
		}
		time.Sleep(50 * time.Millisecond)
	}
	_ = syscall.Kill(pid, syscall.SIGKILL)
	t.Fatalf("grandchild %d survived the timeout", pid)
}
