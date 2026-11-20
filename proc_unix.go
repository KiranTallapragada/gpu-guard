//go:build unix

package main

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

// setProcessGroup starts the command in a new process group and, on cancellation, kills the
// whole group (the shell and every process it started).
func setProcessGroup(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	c.Cancel = func() error {
		if c.Process == nil {
			return nil
		}
		if err := syscall.Kill(-c.Process.Pid, syscall.SIGKILL); errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone // already exited: not a failure
		} else {
			return err
		}
	}
}

// cleanupGroup kills anything the command left running in its process group after the shell
// exited (for example a stray background job). Services should be restarted via systemd or Docker.
func cleanupGroup(c *exec.Cmd) {
	if c.Process != nil {
		_ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
	}
}
