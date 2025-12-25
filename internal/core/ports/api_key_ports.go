package ports

import (
	"github.com/AutoCookies/pomai-cache/internal/core/models"
)

// APIKeyRepository defines methods for interacting with the database
type APIKeyRepository interface {
	CreateAPIKey(apiKey *models.APIKeyModel) error
	GetAPIKeyByID(apiKeyID string) (models.APIKeyModel, error)
	GetAPIKeyByKey(key string) (models.APIKeyModel, error)
	DeactivateAPIKey(apiKeyID string) error
	ListAPIKeysByTenant(tenantID string) ([]models.APIKeyModel, error)
	UpdateAPIKeySecret(apiKeyID string, newHashedKey string) (models.APIKeyModel, error)
	MarkSecretShown(apiKeyID string) error
}

// APIKeyService defines logic for managing API keys
type APIKeyService interface {
	// GenerateAPIKey returns (model, rawSecret, error). rawSecret is only returned once.
	GenerateAPIKey(tenantID string, expiryDays int) (models.APIKeyModel, string, error)
	ValidateAPIKey(rawKey string) (bool, string, error) // returns (ok, tenantID, error)

	// New methods
	ListAPIKeys(tenantID string) ([]models.APIKeyModel, error)
	DeleteAPIKey(apiKeyID string) error
	// RotateAPIKey returns updated model and the new raw secret (to show to user)
	RotateAPIKey(apiKeyID string) (models.APIKeyModel, string, error)
}
