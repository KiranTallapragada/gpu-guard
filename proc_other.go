//go:build !unix

package main

import "os/exec"

// On non-Unix systems only the direct child is stopped on cancellation.
func setProcessGroup(c *exec.Cmd) {}
func cleanupGroup(c *exec.Cmd)    {}
