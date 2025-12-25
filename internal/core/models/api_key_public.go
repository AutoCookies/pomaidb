// internal/core/models/api_key_public.go
package models

import "time"

// APIKeyPublic là dạng trả về cho client — KHÔNG chứa hashed key
type APIKeyPublic struct {
	ID          string    `json:"id"`
	TenantID    string    `json:"tenant_id"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	IsActive    bool      `json:"is_active"`
	UpdatedAt   time.Time `json:"updated_at"`
	SecretShown bool      `json:"secret_shown"`
}

// ToPublic chuyển APIKeyModel -> APIKeyPublic
func (m *APIKeyModel) ToPublic() APIKeyPublic {
	return APIKeyPublic{
		ID:          m.ID,
		TenantID:    m.TenantID,
		CreatedAt:   m.CreatedAt,
		ExpiresAt:   m.ExpiresAt,
		IsActive:    m.IsActive,
		UpdatedAt:   m.UpdatedAt,
		SecretShown: m.SecretShown,
	}
}
