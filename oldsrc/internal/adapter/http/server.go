// File: internal/adapter/http/server.go
package http

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/adapter/http/handlers"
	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
	"github.com/gorilla/mux"
)

// ============================================================================
// SERVER
// ============================================================================

// Server represents the HTTP server
type Server struct {
	tenants    *tenants.Manager
	router     *mux.Router
	handlers   *handlers.HTTPHandlers
	httpServer *http.Server
	config     *ServerConfig
}

// ServerConfig configures the HTTP server
type ServerConfig struct {
	Address         string
	Port            int
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	IdleTimeout     time.Duration
	ShutdownTimeout time.Duration
	EnableCORS      bool
	EnableMetrics   bool
	EnableDebug     bool
}

// DefaultServerConfig returns default server configuration
func DefaultServerConfig() *ServerConfig {
	return &ServerConfig{
		Address:         "0.0.0.0",
		Port:            8080,
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    30 * time.Second,
		IdleTimeout:     120 * time.Second,
		ShutdownTimeout: 30 * time.Second,
		EnableCORS:      true,
		EnableMetrics:   true,
		EnableDebug:     false,
	}
}

// ============================================================================
// CONSTRUCTOR
// ============================================================================

// NewServer creates a new HTTP server with default config
func NewServer(tm *tenants.Manager) *Server {
	return NewServerWithConfig(tm, DefaultServerConfig())
}

// NewServerWithConfig creates a new HTTP server with custom config
func NewServerWithConfig(tm *tenants.Manager, config *ServerConfig) *Server {
	s := &Server{
		tenants:  tm,
		router:   mux.NewRouter(),
		handlers: handlers.NewHTTPHandlers(tm),
		config:   config,
	}

	// Setup routes
	s.setupRoutes()

	// Setup HTTP server
	addr := fmt.Sprintf("%s:%d", config.Address, config.Port)
	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      s.buildMiddlewareChain(),
		ReadTimeout:  config.ReadTimeout,
		WriteTimeout: config.WriteTimeout,
		IdleTimeout:  config.IdleTimeout,
	}

	return s
}

// ============================================================================
// MIDDLEWARE CHAIN
// ============================================================================

// buildMiddlewareChain builds the middleware chain
func (s *Server) buildMiddlewareChain() http.Handler {
	var handler http.Handler = s.router

	// Apply middleware in reverse order (last applied = first executed)

	// 1. Recovery (outermost)
	handler = RecoveryMiddleware(handler)

	// 2. Logging
	handler = LoggingMiddleware(handler)

	// 3. Request ID
	handler = RequestIDMiddleware(handler)

	// 4. Security headers
	handler = SecurityHeadersMiddleware(handler)

	// 5. CORS (if enabled)
	if s.config.EnableCORS {
		handler = CorsMiddleware(handler)
	}

	return handler
}

// ============================================================================
// PUBLIC METHODS
// ============================================================================

// Router returns the HTTP handler
func (s *Server) Router() http.Handler {
	return s.buildMiddlewareChain()
}

// ListenAndServe starts the HTTP server
func (s *Server) ListenAndServe() error {
	log.Printf("[HTTP] Starting server on %s", s.httpServer.Addr)
	log.Printf("[HTTP] Read timeout:   %v", s.config.ReadTimeout)
	log.Printf("[HTTP] Write timeout:   %v", s.config.WriteTimeout)
	log.Printf("[HTTP] Idle timeout: %v", s.config.IdleTimeout)

	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	log.Printf("[HTTP] Shutting down server...")

	// Create timeout context if none provided
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), s.config.ShutdownTimeout)
		defer cancel()
	}

	// Shutdown HTTP server
	if err := s.httpServer.Shutdown(ctx); err != nil {
		return fmt.Errorf("server shutdown error: %w", err)
	}

	log.Printf("[HTTP] Server shutdown complete")
	return nil
}

// GetConfig returns server configuration
func (s *Server) GetConfig() *ServerConfig {
	return s.config
}

// GetHandlers returns HTTP handlers
func (s *Server) GetHandlers() *handlers.HTTPHandlers {
	return s.handlers
}

// ============================================================================
// HELPER METHODS
// ============================================================================

// GetAddr returns the server address
func (s *Server) GetAddr() string {
	return s.httpServer.Addr
}

// IsRunning checks if server is running
func (s *Server) IsRunning() bool {
	return s.httpServer != nil
}
