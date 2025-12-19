package httpadapter

import (
	"context"
	"net/http"
	"strings"
)

type contextKey string

const (
	ctxTenantKey contextKey = "tenantID"
	ctxUserIDKey string     = "userID" // Key string để khớp với logic HandleMe cũ
)

func (s *Server) setupRoutes() {
	// API v1 subrouter
	api := s.router.PathPrefix("/v1").Subrouter()

	// --- AUTH ROUTES ---
	auth := s.router.PathPrefix("/auth").Subrouter()

	auth.HandleFunc("/signup", s.authHandler.HandleSignup).Methods("POST")
	auth.HandleFunc("/signin", s.authHandler.HandleLogin).Methods("POST")
	auth.HandleFunc("/verify-email", s.authHandler.HandleVerifyEmail).Methods("POST")
	auth.HandleFunc("/resend-verification", s.authHandler.HandleResendVerification).Methods("POST")
	auth.HandleFunc("/refresh", s.authHandler.HandleRefresh).Methods("POST")
	auth.HandleFunc("/signout", s.authHandler.HandleSignOut).Methods("POST")

	auth.HandleFunc("/me", s.authMiddleware(s.authHandler.HandleMe)).Methods("GET")

	// --- API_KEY ROUTES ---
	apiKeys := s.router.PathPrefix("/api-key").Subrouter()

	// Create / Generate
	apiKeys.HandleFunc("/generate", s.authMiddleware(s.apiKeyHandler.HandleGenerate)).Methods("POST")
	apiKeys.HandleFunc("/create", s.authMiddleware(s.apiKeyHandler.HandleGenerate)).Methods("POST")

	// List keys
	apiKeys.HandleFunc("/list", s.authMiddleware(s.apiKeyHandler.HandleList)).Methods("GET")

	// Delete (soft-delete / deactivate)
	apiKeys.HandleFunc("/delete", s.authMiddleware(s.apiKeyHandler.HandleDelete)).Methods("DELETE")

	// Rotate
	apiKeys.HandleFunc("/rotate", s.authMiddleware(s.apiKeyHandler.HandleRotate)).Methods("POST")

	// Validate
	apiKeys.HandleFunc("/validate", s.authMiddleware(s.apiKeyHandler.HandleValidate)).Methods("GET")

	// --- CACHE ROUTES (Protected) ---
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handlePut)).Methods("PUT")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleGet)).Methods("GET")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleDelete)).Methods("DELETE")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleHead)).Methods("HEAD")

	// Stats
	api.HandleFunc("/stats", s.authMiddleware(s.handleStats)).Methods("GET")

	s.router.HandleFunc("/health", s.handleHealth).Methods("GET")
}

// Middleware: authMiddleware xác thực JWT Token
func (s *Server) authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.requireAuth {
			// Nếu tắt auth mode (dev), cho qua với default user
			ctx := context.WithValue(r.Context(), ctxTenantKey, "default")
			ctx = context.WithValue(ctx, ctxUserIDKey, "default")
			next(w, r.WithContext(ctx))
			return
		}

		// 1. Lấy token từ Header hoặc Cookie
		tokenString := ""

		// 1a. Check Authorization Header (Bearer <token>)
		authHeader := r.Header.Get("Authorization")
		if len(authHeader) > 7 && strings.ToUpper(authHeader[:7]) == "BEARER " {
			tokenString = authHeader[7:]
		}

		// 1b. Fallback: Check Cookie (nếu header không có)
		if tokenString == "" {
			cookie, err := r.Cookie("accessToken")
			if err == nil {
				tokenString = cookie.Value
			}
		}

		// 2. Nếu không tìm thấy token -> 401
		if tokenString == "" {
			http.Error(w, "unauthorized: missing token", http.StatusUnauthorized)
			return
		}

		// 3. Verify Token
		payload, err := s.tokenMaker.VerifyToken(tokenString)
		if err != nil {
			http.Error(w, "unauthorized: invalid token", http.StatusUnauthorized)
			return
		}

		// 4. Token ngon -> Nhét UserID vào Context
		// ctxUserIDKey ("userID") để HandleMe dùng
		// ctxTenantKey ("tenantID") để Cache Engine dùng (Logic: mỗi user là 1 tenant)
		ctx := context.WithValue(r.Context(), ctxUserIDKey, payload.UserID)
		ctx = context.WithValue(ctx, ctxTenantKey, payload.UserID)

		next(w, r.WithContext(ctx))
	}
}

// Helper lấy tenantID (dành cho các file handler cũ nếu cần)
func tenantFromContext(ctx context.Context) string {
	if v := ctx.Value(ctxTenantKey); v != nil {
		if t, ok := v.(string); ok {
			return t
		}
	}
	return "default"
}
