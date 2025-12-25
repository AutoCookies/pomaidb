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
		// 1. Lấy Origin từ request của người dùng
		origin := r.Header.Get("Origin")

		// 2. Dynamic Reflection: Set lại đúng cái Origin đó vào Header trả về
		// Điều này giúp vượt qua lỗi "Wildcard * not allowed with credentials"
		if origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		} else {
			// Nếu không có Origin (ví dụ gọi từ SDK/Postman), thì để *
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}

		// 3. [QUAN TRỌNG] Cho phép gửi kèm Credentials (Cookies, Auth Headers)
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		// 4. Các Methods cho phép
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, HEAD")

		// 5. Các Headers cho phép
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, Origin, Accept, X-Requested-With")

		// 6. Xử lý Preflight (OPTIONS)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
