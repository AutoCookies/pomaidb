package httpadapter

import (
	"encoding/json"
	"net/http"

	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

// APIKeyHandler struct remains
type APIKeyHandler struct {
	service ports.APIKeyService
}

func NewAPIKeyHandler(service ports.APIKeyService) *APIKeyHandler {
	return &APIKeyHandler{service: service}
}

// HandleGenerate giữ nguyên (tên endpoint có thể là /generate hoặc /create)
func (h *APIKeyHandler) HandleGenerate(w http.ResponseWriter, r *http.Request) {
	// optional: accept tenant & expiry from body or query
	var req struct {
		TenantID   string `json:"tenantId"`
		ExpiryDays int    `json:"expiryDays"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	tenantID := req.TenantID
	if tenantID == "" {
		// fallback: try context tenant
		if t, ok := r.Context().Value(ctxTenantKey).(string); ok {
			tenantID = t
		}
	}
	if tenantID == "" {
		tenantID = "default"
	}
	if req.ExpiryDays <= 0 {
		req.ExpiryDays = 365 * 10 // default long expiry if not provided
	}

	apiKey, err := h.service.GenerateAPIKey(tenantID, req.ExpiryDays)
	if err != nil {
		http.Error(w, "Could not generate API Key: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Return created key model; do NOT expose secret if you don't want to.
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"key":     apiKey,
		"message": "created",
	})
}

// HandleList trả về danh sách keys cho tenant hiện tại
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

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"keys":    keys,
		"message": "ok",
	})
}

// HandleDelete chuyển key thành inactive (soft delete)
func (h *APIKeyHandler) HandleDelete(w http.ResponseWriter, r *http.Request) {
	// try to read id from URL query or body
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
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "deleted",
	})
}

// HandleRotate tạo secret mới cho keyId và trả về record đã update (và new secret nếu muốn)
func (h *APIKeyHandler) HandleRotate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		KeyID string `json:"keyId"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	if req.KeyID == "" {
		http.Error(w, "Missing keyId", http.StatusBadRequest)
		return
	}

	updated, err := h.service.RotateAPIKey(req.KeyID)
	if err != nil {
		http.Error(w, "Could not rotate API Key: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Note: if you want to return the new secret value to the caller,
	// ensure the models.APIKeyModel contains the secret field and you are comfortable exposing it.
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"key":     updated,
		"message": "rotated",
	})
}

// HandleValidate giữ nguyên (đổi key name response shape nếu cần)
func (h *APIKeyHandler) HandleValidate(w http.ResponseWriter, r *http.Request) {
	apiKey := r.URL.Query().Get("key")

	isValid, err := h.service.ValidateAPIKey(apiKey)
	if err != nil {
		http.Error(w, "API Key is invalid: "+err.Error(), http.StatusUnauthorized)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"is_valid": isValid})
}
