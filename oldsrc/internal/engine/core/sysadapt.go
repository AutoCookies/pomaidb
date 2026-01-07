package core

import (
	"log"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/metrics"
)

var (
	OffHeapReporter func() int64
)

func ApplySystemAdaptive() int64 {
	memLimit := detectCgroupMemoryLimit()
	numCPU := runtime.NumCPU()
	if memLimit > 0 {
		perCore := memLimit / int64(max(1, numCPU))
		switch {
		case perCore < (256 << 20):
			newProcs := max(1, numCPU/2)
			runtime.GOMAXPROCS(newProcs)
			log.Printf("[SYSADAPT] Container limit detected. Low memory per core (%s), throttling GOMAXPROCS=%d", humanBytes(perCore), newProcs)
		default:
			runtime.GOMAXPROCS(numCPU)
		}

		off := int64(0)
		if OffHeapReporter != nil {
			off = OffHeapReporter()
			if off < 0 {
				off = 0
			}
		}

		applied := memLimit - off

		if applied < (128 << 20) {
			applied = memLimit
		}
		if applied > math.MaxInt64 {
			applied = math.MaxInt64
		}

		if trySetMemoryLimit(int64(applied)) {
			log.Printf("[SYSADAPT] Applied Go Memory Limit: %s (Container Limit: %s, OffHeap: %s)",
				humanBytes(applied), humanBytes(memLimit), humanBytes(off))
		}

		metrics.SetAppliedMemoryLimit(applied)
		metrics.SetOffheapBytes(off)

		switch {
		case memLimit < (512 << 20):
			debug.SetGCPercent(20)
		case memLimit < (2 << 30):
			debug.SetGCPercent(50)
		default:
			debug.SetGCPercent(100)
		}

		time.Sleep(5 * time.Millisecond)
		return int64(applied)
	}

	metrics.SetAppliedMemoryLimit(0)
	runtime.GOMAXPROCS(numCPU)
	debug.SetGCPercent(100)
	debug.SetMemoryLimit(-1)

	log.Printf("[SYSADAPT] No Container memory limit detected. Running in UNLIMITED mode.")
	return 0
}

func detectCgroupMemoryLimit() int64 {
	if b, err := os.ReadFile("/sys/fs/cgroup/memory.max"); err == nil {
		s := strings.TrimSpace(string(b))
		if s != "" && s != "max" {
			if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
				return v
			}
		}
	}

	if b, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes"); err == nil {
		s := strings.TrimSpace(string(b))
		if s != "" {
			if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
				if v > (1 << 60) { // Số quá lớn = Unlimited
					return 0
				}
				return v
			}
		}
	}

	if cgroupPath, ok := findCgroupPath(); ok {
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

func findCgroupPath() (string, bool) {
	b, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", false
	}
	lines := strings.Split(string(b), "\n")
	for _, l := range lines {
		if strings.Contains(l, ":memory:") || strings.Contains(l, ":memory") {
			parts := strings.SplitN(l, ":", 3)
			if len(parts) == 3 {
				path := parts[2]
				if _, err := os.Stat("/sys/fs/cgroup" + path); err == nil {
					return "/sys/fs/cgroup" + path, true
				}
			}
		}
	}
	return "", false
}

func trySetMemoryLimit(limit int64) bool {
	defer func() { _ = recover() }()
	debug.SetMemoryLimit(limit)
	return true
}

func humanBytes(b int64) string {
	if b <= 0 {
		return "0B"
	}
	const unit = 1024
	if b < unit {
		return strconv.FormatInt(b, 10) + "B"
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return strconv.FormatFloat(float64(b)/float64(div), 'f', 1, 64) + " " + string("KMGTPE"[exp]) + "B"
}

func SetOffHeapReporter(fn func() int64) {
	OffHeapReporter = fn
}
