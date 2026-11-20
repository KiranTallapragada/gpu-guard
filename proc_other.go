//go:build !unix

package main

import "os/exec"

// setProcessGroup: on non-Unix systems only the direct child is stopped on cancellation.
func setProcessGroup(c *exec.Cmd) {}
