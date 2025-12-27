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
	apiKeyHandler := NewAPIKeyHandler(apiKeyService)

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

// Router trả về handler đã được bọc middleware CORS
// Đây là chốt chặn quan trọng nhất để fix lỗi trên trình duyệt
func (s *Server) Router() http.Handler {
	return enableCORS(s.router)
}

func enableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Lấy domain từ env (hoặc dùng reflection nhưng phải chuẩn)
		if origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		}

		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, HEAD")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, Origin, Accept, X-Requested-With")

		// QUAN TRỌNG: Với Preflight (OPTIONS), phải trả về 204 hoặc 200 và DỪNG LUÔN
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}
