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
	apiKeys.HandleFunc("/validate", s.apiKeyHandler.HandleValidate).Methods("GET")

	// Discover Server URL by API Key
	apiKeys.HandleFunc("/discover", s.apiKeyHandler.HandleDiscover).Methods("GET", "POST")

	// --- CACHE ROUTES (Protected) ---
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handlePut)).Methods("PUT")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleGet)).Methods("GET")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleDelete)).Methods("DELETE")
	api.HandleFunc("/cache/{key}", s.authMiddleware(s.handleHead)).Methods("HEAD")
	api.HandleFunc("/cache/{key}/incr", s.authMiddleware(s.handleIncr)).Methods("POST")

	// Stats
	api.HandleFunc("/stats", s.authMiddleware(s.handleStats)).Methods("GET")

	s.router.HandleFunc("/health", s.handleHealth).Methods("GET")
}

// Middleware: authMiddleware xác thực JWT Token hoặc API Key
func (s *Server) authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Dev mode: bypass auth and set default tenant/user
		if !s.requireAuth {
			ctx := context.WithValue(r.Context(), ctxTenantKey, "default")
			ctx = context.WithValue(ctx, ctxUserIDKey, "default")
			next(w, r.WithContext(ctx))
			return
		}

		// 1) Try API Key first
		apiKeyHeader := ""
		// header X-API-Key
		if v := r.Header.Get("X-API-Key"); v != "" {
			apiKeyHeader = strings.TrimSpace(v)
		} else {
			// Authorization: ApiKey <key>
			auth := r.Header.Get("Authorization")
			if len(auth) > 7 && strings.HasPrefix(strings.ToLower(auth), "apikey ") {
				apiKeyHeader = strings.TrimSpace(auth[7:])
			}
		}

		if apiKeyHeader != "" {
			// Validate API Key using APIKeyService
			if s.apiKeyService != nil {
				ok, tenantID, err := s.apiKeyService.ValidateAPIKey(apiKeyHeader)
				if err == nil && ok {
					// inject tenant and user (userID = tenantID by convention for API key)
					ctx := context.WithValue(r.Context(), ctxTenantKey, tenantID)
					ctx = context.WithValue(ctx, ctxUserIDKey, tenantID)
					next(w, r.WithContext(ctx))
					return
				}
				// invalid API key
				http.Error(w, "unauthorized: invalid api key", http.StatusUnauthorized)
				return
			}
			// if service not available, fall through to JWT (or reject)
		}

		// 2) Fallback: JWT (Bearer in Authorization header or accessToken cookie)
		tokenString := ""
		authHeader := r.Header.Get("Authorization")
		if len(authHeader) >= 7 && strings.HasPrefix(strings.ToUpper(authHeader), "BEARER ") {
			tokenString = authHeader[7:]
		}
		if tokenString == "" {
			if c, err := r.Cookie("accessToken"); err == nil {
				tokenString = c.Value
			}
		}

		if tokenString == "" {
			http.Error(w, "unauthorized: missing token", http.StatusUnauthorized)
			return
		}

		payload, err := s.tokenMaker.VerifyToken(tokenString)
		if err != nil {
			http.Error(w, "unauthorized: invalid token", http.StatusUnauthorized)
			return
		}

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
