package models

import "time"

// APIKeyModel defines the structure of an API key object
type APIKeyModel struct {
	ID          string    `json:"id"`
	HashedKey   string    `json:"-"`
	TenantID    string    `json:"tenant_id"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	IsActive    bool      `json:"is_active"`
	UpdatedAt   time.Time `json:"updated_at"`
	SecretShown bool      `json:"secret_shown"` // true if secret already displayed to user
}
