package httpadapter

import (
	"net/http"

	"github.com/AutoCookies/pomai-cache/internal/core/ports"
	"github.com/AutoCookies/pomai-cache/internal/engine"
	"github.com/gorilla/mux"
)

// Server wraps handlers for cache engine and auth
type Server struct {
	tenants       *engine.TenantManager
	authHandler   *AuthHandler
	tokenMaker    ports.TokenMaker
	requireAuth   bool
	router        *mux.Router
	apiKeyHandler *APIKeyHandler
	apiKeyService ports.APIKeyService // expose service for middleware use
}

// NewServer creates a new API Server instance
func NewServer(
	tenants *engine.TenantManager,
	authHandler *AuthHandler,
	tokenMaker ports.TokenMaker,
	requireAuth bool,
	apiKeyService ports.APIKeyService,
) *Server {
	apiKeyHandler := NewAPIKeyHandler(apiKeyService) // Khởi tạo APIKeyHandler

	s := &Server{
		tenants:       tenants,
		authHandler:   authHandler,
		tokenMaker:    tokenMaker,
		requireAuth:   requireAuth,
		router:        mux.NewRouter(),
		apiKeyHandler: apiKeyHandler,
		apiKeyService: apiKeyService,
	}

	s.setupRoutes()
	return s
}

func (s *Server) Router() http.Handler {
	return s.router
}
