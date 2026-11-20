//go:build unix

package main

import (
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
		return syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
	}
}
