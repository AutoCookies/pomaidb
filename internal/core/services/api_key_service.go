package services

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"time"

	"golang.org/x/crypto/bcrypt"

	"github.com/AutoCookies/pomai-cache/internal/core/models"
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

const (
	secretBytes = 32
	bcryptCost  = 12
)

type apiKeyService struct {
	repo ports.APIKeyRepository
}

func NewAPIKeyService(repo ports.APIKeyRepository) ports.APIKeyService {
	return &apiKeyService{repo: repo}
}

func generateRawSecret() (string, error) {
	b := make([]byte, secretBytes)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func hashSecret(secret string) (string, error) {
	h, err := bcrypt.GenerateFromPassword([]byte(secret), bcryptCost)
	if err != nil {
		return "", err
	}
	return string(h), nil
}

// GenerateAPIKey creates a new key record and returns raw secret to caller (only once)
func (s *apiKeyService) GenerateAPIKey(tenantID string, expiryDays int) (models.APIKeyModel, string, error) {
	if tenantID == "" {
		return models.APIKeyModel{}, "", errors.New("tenantID required")
	}

	raw, err := generateRawSecret()
	if err != nil {
		return models.APIKeyModel{}, "", err
	}
	hashed, err := hashSecret(raw)
	if err != nil {
		return models.APIKeyModel{}, "", err
	}
	now := time.Now().UTC()
	exp := now.Add(time.Duration(expiryDays) * 24 * time.Hour)

	model := models.APIKeyModel{
		// ID left empty -> repo will fill it via RETURNING
		HashedKey:   hashed,
		TenantID:    tenantID,
		CreatedAt:   now,
		ExpiresAt:   exp,
		IsActive:    true,
		UpdatedAt:   now,
		SecretShown: true, // we will return raw now
	}

	if err := s.repo.CreateAPIKey(&model); err != nil {
		return models.APIKeyModel{}, "", err
	}

	return model, raw, nil
}

// ValidateAPIKey verifies a provided public key "<kid>.<secret>"
func (s *apiKeyService) ValidateAPIKey(rawKey string) (bool, string, error) {
	sep := -1
	for i := 0; i < len(rawKey); i++ {
		if rawKey[i] == '.' {
			sep = i
			break
		}
	}
	if sep <= 0 {
		return false, "", errors.New("invalid key format")
	}
	kid := rawKey[:sep]
	secret := rawKey[sep+1:]

	rec, err := s.repo.GetAPIKeyByID(kid)
	if err != nil {
		return false, "", err
	}
	if !rec.IsActive {
		return false, "", errors.New("key inactive")
	}
	if !rec.ExpiresAt.IsZero() && time.Now().After(rec.ExpiresAt) {
		return false, "", errors.New("key expired")
	}
	// compare bcrypt hash
	if err := bcrypt.CompareHashAndPassword([]byte(rec.HashedKey), []byte(secret)); err != nil {
		return false, "", errors.New("invalid secret")
	}

	// Optionally update last_used_at via repo (not implemented here)
	return true, rec.TenantID, nil
}

func (s *apiKeyService) ListAPIKeys(tenantID string) ([]models.APIKeyModel, error) {
	return s.repo.ListAPIKeysByTenant(tenantID)
}

func (s *apiKeyService) DeleteAPIKey(apiKeyID string) error {
	return s.repo.DeactivateAPIKey(apiKeyID)
}

// RotateAPIKey: generate new secret, hash and update DB; return updated model and raw secret
func (s *apiKeyService) RotateAPIKey(apiKeyID string) (models.APIKeyModel, string, error) {
	newRaw, err := generateRawSecret()
	if err != nil {
		return models.APIKeyModel{}, "", err
	}
	newHashed, err := hashSecret(newRaw)
	if err != nil {
		return models.APIKeyModel{}, "", err
	}
	updated, err := s.repo.UpdateAPIKeySecret(apiKeyID, newHashed)
	if err != nil {
		return models.APIKeyModel{}, "", err
	}
	return updated, newRaw, nil
}
