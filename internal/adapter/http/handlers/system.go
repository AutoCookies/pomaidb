// File: internal/adapter/http/handlers/system.go
package handlers

import (
	"fmt"
	"net/http"
	"runtime"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
)

// ============================================================================
// HEALTH CHECK
// ============================================================================

// HealthResponse represents health check response
type HealthResponse struct {
	Status       string  `json:"status"`
	Timestamp    int64   `json:"timestamp"`
	Uptime       float64 `json:"uptime_seconds"`
	TotalTenants int     `json:"total_tenants"`
	Version      string  `json:"version,omitempty"`
}

var serverStartTime = time.Now()

// HandleHealth checks server health
func (h *HTTPHandlers) HandleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		sendError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET/HEAD allowed")
		return
	}

	tenantIDs := h.Tenants.ListTenants()
	uptime := time.Since(serverStartTime).Seconds()

	health := HealthResponse{
		Status:       "healthy",
		Timestamp:    time.Now().Unix(),
		Uptime:       uptime,
		TotalTenants: len(tenantIDs),
		Version:      "1.0.0", // TODO: Get from build info
	}

	sendJSON(w, http.StatusOK, health)
	logRequest(r, "health check")
}

// ============================================================================
// STATISTICS
// ============================================================================

// StatsResponse represents statistics response
type StatsResponse struct {
	TotalTenants  int                    `json:"total_tenants"`
	ActiveTenants int                    `json:"active_tenants"`
	ServerUptime  float64                `json:"server_uptime_seconds"`
	System        SystemStats            `json:"system"`
	PerTenant     map[string]TenantStats `json:"per_tenant,omitempty"`
	Timestamp     int64                  `json:"timestamp"`
}

// TenantStats represents per-tenant statistics
type TenantStats struct {
	TenantID     string  `json:"tenant_id"`
	Hits         uint64  `json:"hits"`
	Misses       uint64  `json:"misses"`
	Items        int64   `json:"items"`
	Bytes        int64   `json:"bytes"`
	Capacity     int64   `json:"capacity"`
	Evictions    uint64  `json:"evictions"`
	HitRate      float64 `json:"hit_rate_percent"`
	MissRate     float64 `json:"miss_rate_percent"`
	UsagePercent float64 `json:"usage_percent"`
	AvgItemSize  int64   `json:"avg_item_size_bytes"`
}

// SystemStats represents system-level statistics
type SystemStats struct {
	GoVersion    string `json:"go_version"`
	NumGoroutine int    `json:"num_goroutine"`
	NumCPU       int    `json:"num_cpu"`
	MemAlloc     uint64 `json:"mem_alloc_bytes"`
	MemTotal     uint64 `json:"mem_total_bytes"`
	MemSys       uint64 `json:"mem_sys_bytes"`
	NumGC        uint32 `json:"num_gc"`
	GCPauseTotal uint64 `json:"gc_pause_total_ns"`
}

// HandleStats returns detailed statistics
func (h *HTTPHandlers) HandleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET allowed")
		return
	}

	// Check if requesting specific tenant
	tenantIDQuery := r.URL.Query().Get("tenantId")
	if tenantIDQuery == "" {
		tenantIDQuery = r.URL.Query().Get("userId") // Backward compatibility
	}

	if tenantIDQuery != "" {
		h.handleSingleTenantStats(w, r, tenantIDQuery)
		return
	}

	// Return all tenants stats
	h.handleAllTenantsStats(w, r)
}

// handleSingleTenantStats returns stats for a specific tenant
func (h *HTTPHandlers) handleSingleTenantStats(w http.ResponseWriter, r *http.Request, tenantID string) {
	store := h.Tenants.GetStore(tenantID)
	if store == nil {
		sendError(w, r, http.StatusNotFound, "tenant_not_found", "Tenant not found")
		return
	}

	stats := store.Stats()

	response := map[string]interface{}{
		"tenant_id": tenantID,
		"stats":     convertToTenantStats(stats),
		"timestamp": time.Now().Unix(),
	}

	sendJSON(w, http.StatusOK, response)
	logRequest(r, "tenant stats retrieved")
}

// handleAllTenantsStats returns stats for all tenants
func (h *HTTPHandlers) handleAllTenantsStats(w http.ResponseWriter, r *http.Request) {
	tenantIDs := h.Tenants.ListTenants()
	uptime := time.Since(serverStartTime).Seconds()

	// Collect per-tenant stats
	perTenant := make(map[string]TenantStats)
	activeTenants := 0

	for _, tenantID := range tenantIDs {
		store := h.Tenants.GetStore(tenantID)
		if store == nil {
			continue
		}

		stats := store.Stats()
		perTenant[tenantID] = convertToTenantStats(stats)

		if stats.Items > 0 {
			activeTenants++
		}
	}

	// System stats
	systemStats := collectSystemStats()

	// Build response
	response := StatsResponse{
		TotalTenants:  len(tenantIDs),
		ActiveTenants: activeTenants,
		ServerUptime:  uptime,
		System:        systemStats,
		PerTenant:     perTenant,
		Timestamp:     time.Now().Unix(),
	}

	sendJSON(w, http.StatusOK, response)
	logRequest(r, "all stats retrieved")
}

// ============================================================================
// METRICS (Prometheus-style)
// ============================================================================

// HandleMetrics returns Prometheus-style metrics
func (h *HTTPHandlers) HandleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET allowed")
		return
	}

	tenantIDs := h.Tenants.ListTenants()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.WriteHeader(http.StatusOK)

	// Server metrics
	uptime := time.Since(serverStartTime).Seconds()
	w.Write([]byte("# HELP pomai_server_uptime_seconds Server uptime in seconds\n"))
	w.Write([]byte("# TYPE pomai_server_uptime_seconds gauge\n"))
	w.Write([]byte(sprintf("pomai_server_uptime_seconds %. 2f\n", uptime)))

	w.Write([]byte("# HELP pomai_tenants_total Total number of tenants\n"))
	w.Write([]byte("# TYPE pomai_tenants_total gauge\n"))
	w.Write([]byte(sprintf("pomai_tenants_total %d\n", len(tenantIDs))))

	// Per-tenant metrics
	for _, tenantID := range tenantIDs {
		store := h.Tenants.GetStore(tenantID)
		if store == nil {
			continue
		}

		stats := store.Stats()

		// Hits
		w.Write([]byte(sprintf("pomai_cache_hits_total{tenant=\"%s\"} %d\n", tenantID, stats.Hits)))

		// Misses
		w.Write([]byte(sprintf("pomai_cache_misses_total{tenant=\"%s\"} %d\n", tenantID, stats.Misses)))

		// Items
		w.Write([]byte(sprintf("pomai_cache_items{tenant=\"%s\"} %d\n", tenantID, stats.Items)))

		// Bytes
		w.Write([]byte(sprintf("pomai_cache_bytes{tenant=\"%s\"} %d\n", tenantID, stats.Bytes)))

		// Evictions
		w.Write([]byte(sprintf("pomai_cache_evictions_total{tenant=\"%s\"} %d\n", tenantID, stats.Evictions)))

		// Hit rate
		w.Write([]byte(sprintf("pomai_cache_hit_rate{tenant=\"%s\"} %.2f\n", tenantID, stats.HitRate/100)))
	}

	// System metrics
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	w.Write([]byte("# HELP pomai_go_goroutines Number of goroutines\n"))
	w.Write([]byte("# TYPE pomai_go_goroutines gauge\n"))
	w.Write([]byte(sprintf("pomai_go_goroutines %d\n", runtime.NumGoroutine())))

	w.Write([]byte("# HELP pomai_go_memory_alloc_bytes Allocated memory in bytes\n"))
	w.Write([]byte("# TYPE pomai_go_memory_alloc_bytes gauge\n"))
	w.Write([]byte(sprintf("pomai_go_memory_alloc_bytes %d\n", memStats.Alloc)))

	logRequest(r, "metrics retrieved")
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// convertToTenantStats converts core.Stats to TenantStats
func convertToTenantStats(stats common.Stats) TenantStats {
	return TenantStats{
		TenantID:     stats.TenantID,
		Hits:         stats.Hits,
		Misses:       stats.Misses,
		Items:        stats.Items,
		Bytes:        stats.Bytes,
		Capacity:     stats.Capacity,
		Evictions:    stats.Evictions,
		HitRate:      stats.HitRate,
		MissRate:     stats.MissRate,
		UsagePercent: stats.UsagePercent,
		AvgItemSize:  stats.AvgItemSize,
	}
}

// collectSystemStats collects system-level statistics
func collectSystemStats() SystemStats {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return SystemStats{
		GoVersion:    runtime.Version(),
		NumGoroutine: runtime.NumGoroutine(),
		NumCPU:       runtime.NumCPU(),
		MemAlloc:     memStats.Alloc,
		MemTotal:     memStats.TotalAlloc,
		MemSys:       memStats.Sys,
		NumGC:        memStats.NumGC,
		GCPauseTotal: memStats.PauseTotalNs,
	}
}

// sprintf is a simple helper (avoids fmt import in hot path)
func sprintf(format string, args ...interface{}) string {
	// Simple implementation for metrics
	// In production, use fmt.Sprintf or a faster alternative
	return fmt.Sprintf(format, args...)
}

// ============================================================================
// READINESS & LIVENESS (Kubernetes)
// ============================================================================

// HandleReadiness checks if server is ready to serve traffic
func (h *HTTPHandlers) HandleReadiness(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		sendError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET/HEAD allowed")
		return
	}

	// Check if at least one tenant is available
	tenantIDs := h.Tenants.ListTenants()
	if len(tenantIDs) == 0 {
		sendError(w, r, http.StatusServiceUnavailable, "not_ready", "No tenants available")
		return
	}

	sendJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "ready",
		"timestamp": time.Now().Unix(),
	})
}

// HandleLiveness checks if server is alive
func (h *HTTPHandlers) HandleLiveness(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		sendError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET/HEAD allowed")
		return
	}

	sendJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "alive",
		"timestamp": time.Now().Unix(),
	})
}
