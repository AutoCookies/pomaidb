package core

import (
	"bufio"
	"bytes"
	"io"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// ApplySystemAdaptive inspects the runtime / cgroup environment and applies
// conservative defaults so the process behaves well on small machines / docker.
// It intentionally does NOT depend on main.Config — it auto-detects and tunes:
//   - GOMAXPROCS
//   - debug.SetMemoryLimit (if cgroup limit detected)
//   - debug.SetGCPercent (lower GC aggressiveness on small memory)
func ApplySystemAdaptive() {
	numCPU := runtime.NumCPU()

	// Detect memory limit (try cgroup v2/v1, fallback to /proc/meminfo)
	memLimit := detectCgroupMemoryLimit()
	if memLimit <= 0 {
		memLimit = detectTotalMemoryFromProc()
	}

	// Heuristics for GOMAXPROCS:
	// - If memory is very low per core, reduce parallelism to avoid OOM
	// - Otherwise let GOMAXPROCS==NumCPU
	if memLimit > 0 {
		perCore := memLimit / int64(max(1, numCPU))
		// thresholds tuned for safety on small machines:
		switch {
		case perCore < (256 << 20): // <256MB/core -> cap at 1/2 cores (min 1)
			newProcs := max(1, numCPU/2)
			runtime.GOMAXPROCS(newProcs)
		case perCore < (512 << 20): // <512MB/core -> cap at numCPU-1
			newProcs := max(1, numCPU-1)
			runtime.GOMAXPROCS(newProcs)
		default:
			runtime.GOMAXPROCS(numCPU)
		}
		// Apply memory limit to Go runtime if we detected a limit and API available
		// Use a slightly smaller limit to leave headroom for OS and other processes.
		// Only do this on Go versions that support SetMemoryLimit; silence if fails.
		limitForGo := uint64(memLimit)
		// reserve 5% headroom
		headroom := uint64(limitForGo / 20)
		if headroom < (32 << 20) {
			headroom = 32 << 20
		}
		_ = trySetMemoryLimit(limitForGo - headroom)
	} else {
		// No detection -> default to full CPU
		runtime.GOMAXPROCS(numCPU)
	}

	// GC tuning based on absolute memory available
	if memLimit > 0 {
		switch {
		case memLimit < (512 << 20): // <512MB
			debug.SetGCPercent(20)
		case memLimit < (2 << 30): // <2GB
			debug.SetGCPercent(50)
		default:
			// default GC
			debug.SetGCPercent(-1)
		}
	} else {
		// if no mem info, leave as-is (main may set GOGC)
	}

	// Small sleep to allow logs in main to reflect changes (non-critical)
	time.Sleep(5 * time.Millisecond)
}

// detectCgroupMemoryLimit tries cgroup v2 then v1 to obtain a memory limit in bytes.
// Returns 0 if none found or not running on Linux.
func detectCgroupMemoryLimit() int64 {
	// cgroup v2: /sys/fs/cgroup/memory.max
	if b, err := os.ReadFile("/sys/fs/cgroup/memory.max"); err == nil {
		s := strings.TrimSpace(string(b))
		if s != "" && s != "max" {
			if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
				return v
			}
		}
	}

	// cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
	if b, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes"); err == nil {
		s := strings.TrimSpace(string(b))
		if s != "" {
			if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
				// Some systems report huge value for "unlimited" (like 9223372036854771712).
				// Treat very large numbers as unlimited.
				if v > (1 << 60) {
					return 0
				}
				return v
			}
		}
	}

	// Try to parse /proc/self/cgroup for cgroup path and read memory.max in that path (kubernetes/docker)
	if cgroupPath, ok := findCgroupPath(); ok {
		// Try v2 memory.max under cgroupPath
		tryPaths := []string{
			cgroupPath + "/memory.max",
			cgroupPath + "/memory/memory.limit_in_bytes",
		}
		for _, p := range tryPaths {
			if b, err := os.ReadFile(p); err == nil {
				s := strings.TrimSpace(string(b))
				if s != "" && s != "max" {
					if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
						if v > (1 << 60) {
							return 0
						}
						return v
					}
				}
			}
		}
	}

	return 0
}

// detectTotalMemoryFromProc is best-effort fallback: read /proc/meminfo MemTotal (Linux).
func detectTotalMemoryFromProc() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()

	var buf bytes.Buffer
	_, _ = io.CopyN(&buf, f, 4096)
	scanner := bufio.NewScanner(&buf)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				if kb, err := strconv.ParseInt(parts[1], 10, 64); err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 0
}

// findCgroupPath tries to parse /proc/self/cgroup and return possible cgroup fs path
func findCgroupPath() (string, bool) {
	b, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", false
	}
	// Look for line that contains "memory"
	lines := strings.Split(string(b), "\n")
	for _, l := range lines {
		if strings.Contains(l, ":memory:") || strings.Contains(l, ":memory") {
			parts := strings.SplitN(l, ":", 3)
			if len(parts) == 3 {
				path := parts[2]
				// Common mount point
				if _, err := os.Stat("/sys/fs/cgroup" + path); err == nil {
					return "/sys/fs/cgroup" + path, true
				}
			}
		}
	}
	return "", false
}

// trySetMemoryLimit wraps debug.SetMemoryLimit but recovers from panic on older go versions
func trySetMemoryLimit(limit uint64) (ok bool) {
	// debug.SetMemoryLimit exists in go1.19+. Call safely.
	defer func() {
		_ = recover()
	}()

	// Clamp to MaxInt64 to match debug.SetMemoryLimit signature (int64)
	if limit > uint64(math.MaxInt64) {
		limit = uint64(math.MaxInt64)
	}
	debug.SetMemoryLimit(int64(limit))
	return true
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
