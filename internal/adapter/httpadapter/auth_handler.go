package httpadapter

import (
	"encoding/json"
	"net/http"
	"os"

	"github.com/AutoCookies/pomai-cache/internal/core/models"
	"github.com/AutoCookies/pomai-cache/internal/core/services"
)

// AuthHandler wraps the AuthService
type AuthHandler struct {
	service *services.AuthService
}

// NewAuthHandler constructor
func NewAuthHandler(service *services.AuthService) *AuthHandler {
	return &AuthHandler{service: service}
}

// --- COOKIE HELPERS ---

func setAuthCookies(w http.ResponseWriter, accessToken, refreshToken string) {
	isProd := os.Getenv("ENV") == "production"
	// Nếu bạn chạy frontend và backend trên origins khác trong dev,
	// modern browsers yêu cầu SameSite=None và Secure=true => cần HTTPS.
	// Thay vào đó, ở dev ta dùng SameSite=Lax và Secure=false (nếu bạn proxy requests)
	sameSite := http.SameSiteNoneMode
	secure := isProd

	if !isProd {
		// DEV: nếu bạn không chạy HTTPS, set SameSite Lax để cookie không bị browser từ chối.
		// Lưu ý: nếu frontend gọi API cross-site (different origin) thì Lax có thể không gửi cookie cho XHR POST.
		sameSite = http.SameSiteLaxMode
		secure = false
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "accessToken",
		Value:    accessToken,
		HttpOnly: true,
		Secure:   secure,
		Path:     "/",
		MaxAge:   15 * 60,
		SameSite: sameSite,
	})
	http.SetCookie(w, &http.Cookie{
		Name:     "refreshToken",
		Value:    refreshToken,
		HttpOnly: true,
		Secure:   secure,
		Path:     "/",
		MaxAge:   7 * 24 * 60 * 60,
		SameSite: sameSite,
	})
}

func clearAuthCookies(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{Name: "accessToken", Value: "", MaxAge: -1, Path: "/"})
	http.SetCookie(w, &http.Cookie{Name: "refreshToken", Value: "", MaxAge: -1, Path: "/"})
}

// --- HANDLERS ---

// HandleSignup (signupController)
func (h *AuthHandler) HandleSignup(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email       string `json:"email"`
		Password    string `json:"password"`
		DisplayName string `json:"displayName"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Email == "" || req.Password == "" {
		http.Error(w, "Email and password required", http.StatusBadRequest)
		return
	}

	params := models.CreateUserParams{
		Email:       req.Email,
		Password:    req.Password,
		DisplayName: req.DisplayName,
	}

	user, err := h.service.Register(r.Context(), params)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"user":             user,
		"verificationSent": true,
	})
}

// HandleVerifyEmail (verifyEmailController)
func (h *AuthHandler) HandleVerifyEmail(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email string `json:"email"`
		Code  string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Email == "" || req.Code == "" {
		http.Error(w, "Email and code required", http.StatusBadRequest)
		return
	}

	resp, err := h.service.VerifyEmail(r.Context(), req.Email, req.Code)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	setAuthCookies(w, resp.AccessToken, resp.RefreshToken)
	json.NewEncoder(w).Encode(resp)
}

// HandleLogin (signinController)
func (h *AuthHandler) HandleLogin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	resp, err := h.service.Login(r.Context(), req.Email, req.Password)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	setAuthCookies(w, resp.AccessToken, resp.RefreshToken)
	json.NewEncoder(w).Encode(resp)
}

// HandleRefresh (refreshController)
func (h *AuthHandler) HandleRefresh(w http.ResponseWriter, r *http.Request) {
	refreshToken := ""
	if c, err := r.Cookie("refreshToken"); err == nil {
		refreshToken = c.Value
	}

	if refreshToken == "" {
		var req struct {
			RefreshToken string `json:"refreshToken"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		refreshToken = req.RefreshToken
	}

	if refreshToken == "" {
		http.Error(w, "Missing refresh token", http.StatusBadRequest)
		return
	}

	resp, err := h.service.RefreshToken(r.Context(), refreshToken)
	if err != nil {
		clearAuthCookies(w)
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	setAuthCookies(w, resp.AccessToken, resp.RefreshToken)
	json.NewEncoder(w).Encode(map[string]string{
		"accessToken":  resp.AccessToken,
		"refreshToken": resp.RefreshToken,
	})
}

// HandleSignOut (signoutController)
func (h *AuthHandler) HandleSignOut(w http.ResponseWriter, r *http.Request) {
	refreshToken := ""
	if c, err := r.Cookie("refreshToken"); err == nil {
		refreshToken = c.Value
	} else {
		var req struct {
			RefreshToken string `json:"refreshToken"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		refreshToken = req.RefreshToken
	}

	_ = h.service.SignOut(r.Context(), refreshToken)

	clearAuthCookies(w)
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"ok": true}`))
}

// HandleResendVerification (resendVerificationController)
func (h *AuthHandler) HandleResendVerification(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email string `json:"email"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Email == "" {
		http.Error(w, "Email required", http.StatusBadRequest)
		return
	}

	err := h.service.ResendVerification(r.Context(), req.Email)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":               true,
		"verificationSent": true,
	})
}

// HandleMe (meController)
func (h *AuthHandler) HandleMe(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value("userID").(string)
	if !ok || userID == "" {
		if tID, ok := r.Context().Value("tenantID").(string); ok && tID != "default" {
			userID = tID
		} else {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
	}

	user, err := h.service.GetMe(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if user == nil {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(user)
}
