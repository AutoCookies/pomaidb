// File: internal/adapter/http/middleware.go
package http

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
)

// ============================================================================
// CONTEXT KEYS
// ============================================================================

type contextKey string

const (
	// ContextKeyTenantID is the context key for tenant ID
	ContextKeyTenantID contextKey = "tenantID"

	// ContextKeyRequestID is the context key for request ID
	ContextKeyRequestID contextKey = "requestID"

	// ContextKeyStartTime is the context key for request start time
	ContextKeyStartTime contextKey = "startTime"
)

// ============================================================================
// METRICS
// ============================================================================

var (
	totalRequests   atomic.Uint64
	activeRequests  atomic.Int64
	totalErrors     atomic.Uint64
	totalRateLimits atomic.Uint64
)

// GetMiddlewareStats returns middleware statistics
func GetMiddlewareStats() map[string]interface{} {
	return map[string]interface{}{
		"total_requests":    totalRequests.Load(),
		"active_requests":   activeRequests.Load(),
		"total_errors":      totalErrors.Load(),
		"total_rate_limits": totalRateLimits.Load(),
	}
}

// ============================================================================
// CORS MIDDLEWARE
// ============================================================================

// CorsMiddleware handles CORS (Cross-Origin Resource Sharing)
// Development mode: allows requests from any origin
func CorsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Allow specific origin or wildcard
		if origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}

		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE, HEAD, PATCH")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Requested-With, X-Tenant-ID, X-Request-ID")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "3600")

		// Handle preflight requests
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// CorsMiddlewareProduction handles CORS with stricter rules for production
func CorsMiddlewareProduction(allowedOrigins []string) func(http.Handler) http.Handler {
	allowedOriginsMap := make(map[string]bool)
	for _, origin := range allowedOrigins {
		allowedOriginsMap[origin] = true
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			// Check if origin is allowed
			if origin != "" && allowedOriginsMap[origin] {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}

			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Tenant-ID")
			w.Header().Set("Access-Control-Max-Age", "3600")

			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// ============================================================================
// TENANT MIDDLEWARE
// ============================================================================

// TenantMiddleware extracts tenant ID and enforces concurrency limits
func TenantMiddleware(tm *tenants.Manager) func(http.HandlerFunc) http.HandlerFunc {
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			// Extract tenant ID from header
			tenantID := r.Header.Get("X-Tenant-ID")
			if strings.TrimSpace(tenantID) == "" {
				tenantID = "default"
			}

			// Validate tenant ID
			if !isValidTenantID(tenantID) {
				http.Error(w, "invalid tenant ID", http.StatusBadRequest)
				return
			}

			// Acquire tenant slot (rate limiting)
			if !tm.AcquireTenant(tenantID, 3*time.Second) {
				totalRateLimits.Add(1)
				w.Header().Set("Retry-After", "3")
				http.Error(w, "tenant too many concurrent requests", http.StatusTooManyRequests)
				return
			}
			defer tm.ReleaseTenant(tenantID)

			// Inject tenant ID into context
			ctx := context.WithValue(r.Context(), ContextKeyTenantID, tenantID)
			next(w, r.WithContext(ctx))
		}
	}
}

// GetTenantID extracts tenant ID from context
func GetTenantID(ctx context.Context) string {
	if tenantID, ok := ctx.Value(ContextKeyTenantID).(string); ok {
		return tenantID
	}
	return "default"
}

// isValidTenantID validates tenant ID format
func isValidTenantID(tenantID string) bool {
	if len(tenantID) == 0 || len(tenantID) > 64 {
		return false
	}
	// Allow alphanumeric, underscore, hyphen
	for _, ch := range tenantID {
		if !((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
			(ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
			return false
		}
	}
	return true
}

// ============================================================================
// LOGGING MIDDLEWARE
// ============================================================================

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		activeRequests.Add(1)
		totalRequests.Add(1)

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		// Inject start time into context
		ctx := context.WithValue(r.Context(), ContextKeyStartTime, start)

		// Process request
		next.ServeHTTP(wrapped, r.WithContext(ctx))

		// Log after completion
		duration := time.Since(start)
		activeRequests.Add(-1)

		log.Printf("[HTTP] %s %s %d %v %s",
			r.Method,
			r.URL.Path,
			wrapped.statusCode,
			duration,
			r.RemoteAddr,
		)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// ============================================================================
// RECOVERY MIDDLEWARE
// ============================================================================

// RecoveryMiddleware recovers from panics
func RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				totalErrors.Add(1)

				log.Printf("[HTTP] Panic recovered: %v\n%s", err, debug.Stack())

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprintf(w, `{"error":"internal server error","message":"panic recovered"}`)
			}
		}()

		next.ServeHTTP(w, r)
	})
}

// ============================================================================
// REQUEST ID MIDDLEWARE
// ============================================================================

// RequestIDMiddleware adds unique request ID
func RequestIDMiddleware(next http.Handler) http.Handler {
	var requestCounter atomic.Uint64

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if client provided request ID
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			// Generate request ID
			requestID = fmt.Sprintf("%d-%d", time.Now().UnixNano(), requestCounter.Add(1))
		}

		// Add to response header
		w.Header().Set("X-Request-ID", requestID)

		// Inject into context
		ctx := context.WithValue(r.Context(), ContextKeyRequestID, requestID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// GetRequestID extracts request ID from context
func GetRequestID(ctx context.Context) string {
	if requestID, ok := ctx.Value(ContextKeyRequestID).(string); ok {
		return requestID
	}
	return ""
}

// ============================================================================
// TIMEOUT MIDDLEWARE
// ============================================================================

// TimeoutMiddleware adds request timeout
func TimeoutMiddleware(timeout time.Duration) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			defer cancel()

			done := make(chan struct{})
			go func() {
				next.ServeHTTP(w, r.WithContext(ctx))
				close(done)
			}()

			select {
			case <-done:
				// Request completed
			case <-ctx.Done():
				// Timeout
				if ctx.Err() == context.DeadlineExceeded {
					http.Error(w, "request timeout", http.StatusRequestTimeout)
				}
			}
		})
	}
}

// ============================================================================
// SECURITY MIDDLEWARE
// ============================================================================

// SecurityHeadersMiddleware adds security headers
func SecurityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		w.Header().Set("Content-Security-Policy", "default-src 'self'")

		next.ServeHTTP(w, r)
	})
}

// ============================================================================
// RATE LIMIT MIDDLEWARE (Simple)
// ============================================================================

// SimplerRateLimitMiddleware limits requests per IP
func SimpleRateLimitMiddleware(requestsPerSecond int) func(http.Handler) http.Handler {
	type clientInfo struct {
		tokens    int
		lastReset time.Time
	}

	var (
		mu      sync.RWMutex
		clients = make(map[string]*clientInfo)
	)

	// Cleanup old entries every minute
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			mu.Lock()
			now := time.Now()
			for ip, info := range clients {
				if now.Sub(info.lastReset) > 5*time.Minute {
					delete(clients, ip)
				}
			}
			mu.Unlock()
		}
	}()

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := getClientIP(r)
			now := time.Now()

			mu.Lock()
			client, exists := clients[ip]
			if !exists {
				client = &clientInfo{
					tokens:    requestsPerSecond,
					lastReset: now,
				}
				clients[ip] = client
			}

			// Reset tokens if 1 second passed
			if now.Sub(client.lastReset) >= time.Second {
				client.tokens = requestsPerSecond
				client.lastReset = now
			}

			if client.tokens <= 0 {
				mu.Unlock()
				totalRateLimits.Add(1)
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}

			client.tokens--
			mu.Unlock()

			next.ServeHTTP(w, r)
		})
	}
}

// getClientIP extracts client IP from request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		return strings.TrimSpace(ips[0])
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Use RemoteAddr
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}
	return ip
}

// ============================================================================
// COMPRESSION MIDDLEWARE (Optional)
// ============================================================================

// CompressionMiddleware adds gzip compression (placeholder)
// Note: For production, use github.com/NYTimes/gziphandler
func CompressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if client accepts gzip
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			// TODO: Implement gzip compression
			// For now, just pass through
		}
		next.ServeHTTP(w, r)
	})
}
