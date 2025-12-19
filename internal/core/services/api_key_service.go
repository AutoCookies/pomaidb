package services

import (
	"crypto/rand"
	"encoding/hex"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/core/models"
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

type APIKeyService struct {
	repo ports.APIKeyRepository
}

func NewAPIKeyService(repo ports.APIKeyRepository) ports.APIKeyService {
	return &APIKeyService{repo: repo}
}

func (s *APIKeyService) GenerateAPIKey(tenantID string, expiryDays int) (models.APIKeyModel, error) {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)

	newID := hex.EncodeToString(bytes)
	// For the key material we can generate another random sequence
	secretBytes := make([]byte, 32)
	_, _ = rand.Read(secretBytes)
	secret := hex.EncodeToString(secretBytes)

	apiKey := models.APIKeyModel{
		ID:        newID,
		Key:       secret,
		TenantID:  tenantID,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(time.Duration(expiryDays) * 24 * time.Hour),
		IsActive:  true,
		UpdatedAt: time.Now(),
	}

	err := s.repo.CreateAPIKey(apiKey)
	return apiKey, err
}

func (s *APIKeyService) ValidateAPIKey(key string) (bool, error) {
	apiKey, err := s.repo.GetAPIKeyByKey(key)
	if err != nil {
		return false, err
	}

	if !apiKey.IsActive || time.Now().After(apiKey.ExpiresAt) {
		return false, nil
	}
	return true, nil
}

// ListAPIKeys trả về tất cả keys của tenant
func (s *APIKeyService) ListAPIKeys(tenantID string) ([]models.APIKeyModel, error) {
	return s.repo.ListAPIKeysByTenant(tenantID)
}

// DeleteAPIKey hủy kích hoạt key (soft delete)
func (s *APIKeyService) DeleteAPIKey(apiKeyID string) error {
	return s.repo.DeactivateAPIKey(apiKeyID)
}

// RotateAPIKey thay key mới và trả về bản ghi đã update (và mới secret)
func (s *APIKeyService) RotateAPIKey(apiKeyID string) (models.APIKeyModel, error) {
	// generate new secret
	secretBytes := make([]byte, 32)
	_, _ = rand.Read(secretBytes)
	newSecret := hex.EncodeToString(secretBytes)

	updated, err := s.repo.UpdateAPIKeySecret(apiKeyID, newSecret)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return updated, nil
}
