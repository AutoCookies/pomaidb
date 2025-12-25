package httpadapter

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"github.com/AutoCookies/pomai-cache/internal/core/models"
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

// APIKeyHandler handles API key endpoints
type APIKeyHandler struct {
	service ports.APIKeyService
}

type DiscoverResp struct {
	BaseURL  string `json:"base_url"`
	TenantID string `json:"tenant_id,omitempty"`
	Message  string `json:"message,omitempty"`
}

func NewAPIKeyHandler(service ports.APIKeyService) *APIKeyHandler {
	return &APIKeyHandler{service: service}
}

// --- Responses ---
type createResp struct {
	KeyID       string `json:"key_id"`
	Secret      string `json:"secret,omitempty"` // raw secret shown once only
	TenantID    string `json:"tenant_id"`
	ExpiresAt   any    `json:"expires_at,omitempty"`
	SecretShown bool   `json:"secret_shown"`
	Message     string `json:"message"`
}

type rotateResp = createResp

type listResp struct {
	Keys    []models.APIKeyPublic `json:"keys"`
	Message string                `json:"message"`
}

type validateResp struct {
	IsValid  bool   `json:"is_valid"`
	TenantID string `json:"tenant_id,omitempty"`
	Message  string `json:"message,omitempty"`
}

type ackResp struct {
	OK      bool   `json:"ok"`
	Message string `json:"message,omitempty"`
}

// HandleGenerate creates a new API key and returns the raw secret once.
func (h *APIKeyHandler) HandleGenerate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		TenantID   string `json:"tenantId"`
		ExpiryDays int    `json:"expiryDays"`
		Label      string `json:"label"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	tenantID := req.TenantID
	if tenantID == "" {
		if t, ok := r.Context().Value(ctxTenantKey).(string); ok {
			tenantID = t
		}
	}
	if tenantID == "" {
		tenantID = "default"
	}
	if req.ExpiryDays <= 0 {
		req.ExpiryDays = 365 * 10
	}

	model, rawSecret, err := h.service.GenerateAPIKey(tenantID, req.ExpiryDays)
	if err != nil {
		http.Error(w, "Could not generate API Key: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := createResp{
		KeyID:       model.ID,
		Secret:      rawSecret, // show only once
		TenantID:    model.TenantID,
		ExpiresAt:   model.ExpiresAt,
		SecretShown: model.SecretShown,
		Message:     "created",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(resp)
}

// HandleList returns public metadata for API keys for the tenant.
// It uses APIKeyPublic to ensure hashed keys are never leaked.
func (h *APIKeyHandler) HandleList(w http.ResponseWriter, r *http.Request) {
	tenantID := ""
	if t, ok := r.Context().Value(ctxTenantKey).(string); ok {
		tenantID = t
	}
	if tenantID == "" {
		tenantID = r.URL.Query().Get("tenant_id")
	}
	if tenantID == "" {
		tenantID = "default"
	}

	keys, err := h.service.ListAPIKeys(tenantID)
	if err != nil {
		http.Error(w, "Could not list API Keys: "+err.Error(), http.StatusInternalServerError)
		return
	}

	out := make([]models.APIKeyPublic, 0, len(keys))
	for _, k := range keys {
		out = append(out, k.ToPublic())
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(listResp{Keys: out, Message: "ok"})
}

// HandleDelete soft-deactivates the API key by id.
func (h *APIKeyHandler) HandleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		var body struct {
			ID string `json:"id"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		id = body.ID
	}
	if id == "" {
		http.Error(w, "Missing id", http.StatusBadRequest)
		return
	}

	if err := h.service.DeleteAPIKey(id); err != nil {
		http.Error(w, "Could not delete API Key: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{"success": true, "message": "deleted"})
}

// HandleRotate rotates an existing key and returns the new raw secret once.
func (h *APIKeyHandler) HandleRotate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		KeyID string `json:"keyId"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	if req.KeyID == "" {
		http.Error(w, "Missing keyId", http.StatusBadRequest)
		return
	}

	updated, newRaw, err := h.service.RotateAPIKey(req.KeyID)
	if err != nil {
		http.Error(w, "Could not rotate API Key: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := rotateResp{
		KeyID:       updated.ID,
		Secret:      newRaw,
		TenantID:    updated.TenantID,
		ExpiresAt:   updated.ExpiresAt,
		SecretShown: updated.SecretShown,
		Message:     "rotated",
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// HandleValidate validates a provided public key (kid.secret).
func (h *APIKeyHandler) HandleValidate(w http.ResponseWriter, r *http.Request) {
	apiKey := r.URL.Query().Get("key")
	if apiKey == "" {
		http.Error(w, "missing key parameter", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	ok, tenantID, err := h.service.ValidateAPIKey(apiKey)

	// SỬA: Dù lỗi hay không ok, ta vẫn trả về JSON 200 OK
	// để frontend dễ xử lý logic if (valid) ...
	if err != nil || !ok {
		resp := validateResp{
			IsValid: false, // Báo sai
			Message: "invalid key or format",
		}
		json.NewEncoder(w).Encode(resp)
		return
	}

	resp := validateResp{
		IsValid:  true,
		TenantID: tenantID,
		Message:  "ok",
	}
	json.NewEncoder(w).Encode(resp)
}

// --- Optional: acknowledgement endpoint ---
// This will attempt to call a service method MarkSecretShown if the service implements it.
// If not implemented, returns 501 Not Implemented.
type secretAck interface {
	MarkSecretShown(apiKeyID string) error
}

func (h *APIKeyHandler) HandleAckSecretShown(w http.ResponseWriter, r *http.Request) {
	var req struct {
		KeyID string `json:"keyId"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.KeyID == "" {
		http.Error(w, "missing keyId", http.StatusBadRequest)
		return
	}

	if svc, ok := h.service.(secretAck); ok {
		if err := svc.MarkSecretShown(req.KeyID); err != nil {
			http.Error(w, "failed to ack secret: "+err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(ackResp{OK: true, Message: "acknowledged"})
		return
	}

	http.Error(w, "acknowledge not implemented on service", http.StatusNotImplemented)
}

// HandleDiscover validates an API key and returns the assigned base URL for this tenant.
func (h *APIKeyHandler) HandleDiscover(w http.ResponseWriter, r *http.Request) {
	// 1. Lấy API Key từ Header
	apiKey := r.Header.Get("X-API-Key")
	if apiKey == "" {
		auth := r.Header.Get("Authorization")
		if len(auth) > 7 && strings.HasPrefix(strings.ToLower(auth), "apikey ") {
			apiKey = strings.TrimSpace(auth[7:])
		}
	}
	if apiKey == "" {
		http.Error(w, "missing api key", http.StatusUnauthorized)
		return
	}

	// 2. Validate Key
	ok, tenantID, err := h.service.ValidateAPIKey(apiKey)
	if err != nil || !ok {
		http.Error(w, "invalid api key", http.StatusUnauthorized)
		return
	}

	// 3. Lấy URL từ biến môi trường (PUBLIC_URL)
	// Ví dụ bạn set trong .env: PUBLIC_URL=localhost:8080 hoặc http://localhost:8080
	baseURL := os.Getenv("PUBLIC_URL")

	// Fallback nếu quên set biến môi trường (Mặc định về local)
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8080"
	}

	// 4. Chuẩn hóa URL (Quan trọng!)
	// Đảm bảo có http:// hoặc https:// ở đầu
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}

	// Đảm bảo không có dấu / ở cuối
	baseURL = strings.TrimRight(baseURL, "/")

	// Ghép thêm /v1 nếu chưa có (vì SDK trỏ vào subrouter /v1)
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = baseURL + "/v1"
	}

	// Kết quả cuối cùng sẽ luôn chuẩn: "http://localhost:8080/v1"

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(DiscoverResp{
		BaseURL:  baseURL,
		TenantID: tenantID,
		Message:  "ok",
	})
}
