// File: internal/adapter/http/handlers/base.go
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core"
	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
	"github.com/golang/snappy"
)

// ============================================================================
// CONTEXT KEYS
// ============================================================================

type contextKey string

const (
	ContextKeyTenantID  contextKey = "tenantID"
	ContextKeyRequestID contextKey = "requestID"
	ContextKeyStartTime contextKey = "startTime"
)

// ============================================================================
// HANDLER STRUCT
// ============================================================================

// HTTPHandlers contains dependencies for HTTP request handling
// Primary purpose:  Server monitoring and statistics
type HTTPHandlers struct {
	Tenants *tenants.Manager
	config  *HandlerConfig
}

// HandlerConfig configures handler behavior
type HandlerConfig struct {
	MaxValueSize      int64
	EnableCompression bool
	DefaultTTL        time.Duration
	MaxTTL            time.Duration
	EnableDebug       bool
	EnableMetrics     bool
}

// DefaultHandlerConfig returns default configuration
func DefaultHandlerConfig() *HandlerConfig {
	return &HandlerConfig{
		MaxValueSize:      10 * 1024 * 1024, // 10MB
		EnableCompression: true,
		DefaultTTL:        0, // No expiration
		MaxTTL:            24 * time.Hour,
		EnableDebug:       false,
		EnableMetrics:     true,
	}
}

// ============================================================================
// CONSTRUCTOR
// ============================================================================

// NewHTTPHandlers creates HTTP handlers with default config
func NewHTTPHandlers(tm *tenants.Manager) *HTTPHandlers {
	return &HTTPHandlers{
		Tenants: tm,
		config:  DefaultHandlerConfig(),
	}
}

// NewHTTPHandlersWithConfig creates HTTP handlers with custom config
func NewHTTPHandlersWithConfig(tm *tenants.Manager, config *HandlerConfig) *HTTPHandlers {
	return &HTTPHandlers{
		Tenants: tm,
		config:  config,
	}
}

// GetConfig returns handler configuration
func (h *HTTPHandlers) GetConfig() *HandlerConfig {
	return h.config
}

// ============================================================================
// CONTEXT HELPERS
// ============================================================================

// tenantFromContext extracts tenant ID from context
func tenantFromContext(ctx context.Context) string {
	if v := ctx.Value(ContextKeyTenantID); v != nil {
		if t, ok := v.(string); ok {
			return t
		}
	}
	if v := ctx.Value("tenantID"); v != nil {
		if t, ok := v.(string); ok {
			return t
		}
	}
	return "default"
}

// requestIDFromContext extracts request ID from context
func requestIDFromContext(ctx context.Context) string {
	if v := ctx.Value(ContextKeyRequestID); v != nil {
		if t, ok := v.(string); ok {
			return t
		}
	}
	if v := ctx.Value("requestID"); v != nil {
		if t, ok := v.(string); ok {
			return t
		}
	}
	return ""
}

// ============================================================================
// STORE HELPERS
// ============================================================================

// getStoreForRequest retrieves store for current request
func (h *HTTPHandlers) getStoreForRequest(r *http.Request) (*core.Store, string, error) {
	tenantID := tenantFromContext(r.Context())
	store := h.Tenants.GetStore(tenantID)

	if store == nil {
		return nil, tenantID, fmt.Errorf("failed to get store for tenant: %s", tenantID)
	}

	return store, tenantID, nil
}

// ============================================================================
// ENVIRONMENT HELPERS
// ============================================================================

// getenvInt64 reads int64 from environment variable
func getenvInt64(name string, def int64) int64 {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

// getenvBool reads bool from environment variable
func getenvBool(name string, def bool) bool {
	if v := os.Getenv(name); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

// getenvDuration reads duration from environment variable
func getenvDuration(name string, def time.Duration) time.Duration {
	if v := os.Getenv(name); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

// ============================================================================
// COMPRESSION HELPERS (for cached data)
// ============================================================================

// serveValue serves binary data with optional snappy decompression
func serveValue(w http.ResponseWriter, v []byte) {
	if len(v) == 0 {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", "0")
		_, _ = w.Write(v)
		return
	}

	magicByte := v[0]
	payload := v[1:]

	if magicByte == 1 {
		decoded, err := snappy.Decode(nil, payload)
		if err != nil {
			http.Error(w, "decompression error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", strconv.Itoa(len(decoded)))
		w.Header().Set("X-Compression", "snappy")
		_, _ = w.Write(decoded)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	_, _ = w.Write(payload)
}

// ============================================================================
// RESPONSE HELPERS (Primary for stats/monitoring)
// ============================================================================

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Message   string `json:"message"`
	RequestID string `json:"request_id,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

// SuccessResponse represents a success response
type SuccessResponse struct {
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Message   string      `json:"message,omitempty"`
	RequestID string      `json:"request_id,omitempty"`
	Timestamp int64       `json:"timestamp"`
}

// sendError sends JSON error response
func sendError(w http.ResponseWriter, r *http.Request, statusCode int, errorType, message string) {
	requestID := requestIDFromContext(r.Context())

	resp := ErrorResponse{
		Error:     errorType,
		Message:   message,
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("[ERROR] Failed to encode error response: %v", err)
	}
}

// sendSuccess sends JSON success response
func sendSuccess(w http.ResponseWriter, r *http.Request, data interface{}, message string) {
	requestID := requestIDFromContext(r.Context())

	resp := SuccessResponse{
		Success:   true,
		Data:      data,
		Message:   message,
		RequestID: requestID,
		Timestamp: time.Now().Unix(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("[ERROR] Failed to encode success response: %v", err)
	}
}

// sendJSON sends custom JSON response
func sendJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("[ERROR] Failed to encode JSON response: %v", err)
	}
}

// ============================================================================
// LOGGING HELPERS
// ============================================================================

// logRequest logs HTTP request details
func logRequest(r *http.Request, message string) {
	tenantID := tenantFromContext(r.Context())
	requestID := requestIDFromContext(r.Context())

	log.Printf("[HTTP] [%s] [%s] %s %s - %s",
		requestID,
		tenantID,
		r.Method,
		r.URL.Path,
		message,
	)
}

// logError logs error with context
func logError(r *http.Request, err error, message string) {
	tenantID := tenantFromContext(r.Context())
	requestID := requestIDFromContext(r.Context())

	log.Printf("[ERROR] [%s] [%s] %s %s - %s:  %v",
		requestID,
		tenantID,
		r.Method,
		r.URL.Path,
		message,
		err,
	)
}

// ============================================================================
// UTILITY HELPERS
// ============================================================================

// parseQueryInt64 parses int64 from query parameter
func parseQueryInt64(r *http.Request, key string, defaultValue int64) int64 {
	if v := r.URL.Query().Get(key); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return defaultValue
}

// parseQueryBool parses bool from query parameter
func parseQueryBool(r *http.Request, key string, defaultValue bool) bool {
	if v := r.URL.Query().Get(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return defaultValue
}

// getContentType returns content type from request
func getContentType(r *http.Request) string {
	ct := r.Header.Get("Content-Type")
	if ct == "" {
		return "application/octet-stream"
	}
	return ct
}

// isJSONRequest checks if request has JSON content type
func isJSONRequest(r *http.Request) bool {
	ct := r.Header.Get("Content-Type")
	return ct == "application/json" || ct == "application/json; charset=utf-8"
}
