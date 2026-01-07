// File: internal/adapter/http/router.go
package http

import (
	"net/http"

	"encoding/json"

	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ============================================================================
// ROUTE SETUP
// ============================================================================

// setupRoutes configures all HTTP routes
func (s *Server) setupRoutes() {
	h := s.handlers

	// ========================================================================
	// PUBLIC ROUTES (No authentication)
	// ========================================================================

	// Health checks
	s.router.HandleFunc("/health", h.HandleHealth).Methods("GET", "HEAD")
	s.router.HandleFunc("/ready", h.HandleReadiness).Methods("GET", "HEAD")
	s.router.HandleFunc("/live", h.HandleLiveness).Methods("GET", "HEAD")

	// Prometheus metrics (for scraping)
	if s.config.EnableMetrics {
		s.router.Handle("/metrics", promhttp.Handler()).Methods("GET")
	}

	// ========================================================================
	// API V1 ROUTES (With tenant middleware)
	// ========================================================================

	api := s.router.PathPrefix("/v1").Subrouter()

	// Apply tenant middleware to all v1 routes
	api.Use(createTenantMiddlewareAdapter(s.tenants))

	// --- STATISTICS & MONITORING ---
	api.HandleFunc("/stats", h.HandleStats).Methods("GET")

	// ========================================================================
	// API V2 ROUTES (Future)
	// ========================================================================

	// v2 := s.router.PathPrefix("/v2").Subrouter()
	// ...  future routes

	// ========================================================================
	// ROOT & 404
	// ========================================================================

	// Root endpoint
	s.router.HandleFunc("/", handleRoot).Methods("GET")

	// 404 handler
	s.router.NotFoundHandler = http.HandlerFunc(handleNotFound)
}

// ============================================================================
// MIDDLEWARE ADAPTERS
// ============================================================================

// createTenantMiddlewareAdapter creates a mux-compatible middleware
func createTenantMiddlewareAdapter(tm *tenants.Manager) mux.MiddlewareFunc {
	tenantMw := TenantMiddleware(tm)

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Convert HandlerFunc to Handler
			handlerFunc := tenantMw(func(w http.ResponseWriter, r *http.Request) {
				next.ServeHTTP(w, r)
			})
			handlerFunc(w, r)
		})
	}
}

// ============================================================================
// DEFAULT HANDLERS
// ============================================================================

// handleRoot handles root endpoint
func handleRoot(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"service": "Pomai Cache",
		"version": "1.0.0",
		"status":  "running",
		"endpoints": map[string]string{
			"health":  "/health",
			"ready":   "/ready",
			"live":    "/live",
			"stats":   "/v1/stats",
			"metrics": "/metrics",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Simple JSON encoding
	sendJSONSimple(w, response)
}

// handleNotFound handles 404 errors
func handleNotFound(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"error":   "not_found",
		"message": "endpoint not found",
		"path":    r.URL.Path,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)

	sendJSONSimple(w, response)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// sendJSONSimple sends simple JSON response (avoids dependency on handlers package)
func sendJSONSimple(w http.ResponseWriter, data interface{}) {
	// This is a simple implementation
	// In production, use json.NewEncoder(w).Encode(data)

	json.NewEncoder(w).Encode(data)
}
