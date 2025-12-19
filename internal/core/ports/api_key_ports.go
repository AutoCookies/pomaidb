package ports

import (
	"github.com/AutoCookies/pomai-cache/internal/core/models"
)

// APIKeyRepository defines methods for interacting with the database
type APIKeyRepository interface {
	CreateAPIKey(apiKey models.APIKeyModel) error
	GetAPIKeyByID(apiKeyID string) (models.APIKeyModel, error)
	GetAPIKeyByKey(key string) (models.APIKeyModel, error)
	DeactivateAPIKey(apiKeyID string) error

	// New methods
	ListAPIKeysByTenant(tenantID string) ([]models.APIKeyModel, error)
	UpdateAPIKeySecret(apiKeyID string, newKey string) (models.APIKeyModel, error)
}

// APIKeyService defines logic for managing API keys
type APIKeyService interface {
	GenerateAPIKey(tenantID string, expiryDays int) (models.APIKeyModel, error)
	ValidateAPIKey(key string) (bool, error)

	// New methods
	ListAPIKeys(tenantID string) ([]models.APIKeyModel, error)
	DeleteAPIKey(apiKeyID string) error
	RotateAPIKey(apiKeyID string) (models.APIKeyModel, error)
}
