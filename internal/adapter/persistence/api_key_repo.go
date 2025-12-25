package persistence

import (
	"context"
	"errors"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/core/models"
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
	"github.com/jackc/pgx/v5/pgxpool"
)

type APIKeyRepo struct {
	pool *pgxpool.Pool
}

func NewAPIKeyRepo(pool *pgxpool.Pool) ports.APIKeyRepository {
	return &APIKeyRepo{pool: pool}
}

// CreateAPIKey thêm API Key vào cơ sở dữ liệu, lưu hashed_key, secret_shown
func (r *APIKeyRepo) CreateAPIKey(apiKey *models.APIKeyModel) error {
	ctx := context.Background()
	// NOTE: do not pass id, let DB generate it (must have default gen_random_uuid() on column)
	query := `
        INSERT INTO keys_service.api_keys
            (hashed_key, tenant_id, created_at, expires_at, is_active, secret_shown, updated_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        RETURNING id;
    `
	row := r.pool.QueryRow(ctx, query,
		apiKey.HashedKey,
		apiKey.TenantID,
		apiKey.CreatedAt,
		apiKey.ExpiresAt,
		apiKey.IsActive,
		apiKey.SecretShown,
		apiKey.UpdatedAt,
	)

	var id string
	if err := row.Scan(&id); err != nil {
		return err
	}
	apiKey.ID = id
	return nil
}

// GetAPIKeyByID tìm API Key bằng ID
func (r *APIKeyRepo) GetAPIKeyByID(apiKeyID string) (models.APIKeyModel, error) {
	ctx := context.Background()
	query := `SELECT id, hashed_key, tenant_id, created_at, expires_at, is_active, updated_at, coalesce(secret_shown,false)
			  FROM keys_service.api_keys WHERE id = $1`
	row := r.pool.QueryRow(ctx, query, apiKeyID)

	var apiKey models.APIKeyModel
	err := row.Scan(&apiKey.ID, &apiKey.HashedKey, &apiKey.TenantID, &apiKey.CreatedAt, &apiKey.ExpiresAt, &apiKey.IsActive, &apiKey.UpdatedAt, &apiKey.SecretShown)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return apiKey, nil
}

// GetAPIKeyByKey tìm API Key với raw key (format kid.secret)
// We only parse kid and fetch by ID; validation is done in service
func (r *APIKeyRepo) GetAPIKeyByKey(key string) (models.APIKeyModel, error) {
	sep := -1
	for i := 0; i < len(key); i++ {
		if key[i] == '.' {
			sep = i
			break
		}
	}
	if sep <= 0 {
		return models.APIKeyModel{}, errors.New("invalid key format")
	}
	kid := key[:sep]
	return r.GetAPIKeyByID(kid)
}

// DeactivateAPIKey hủy kích hoạt API Key
func (r *APIKeyRepo) DeactivateAPIKey(apiKeyID string) error {
	ctx := context.Background()
	query := `UPDATE keys_service.api_keys SET is_active = FALSE, updated_at = $2 WHERE id = $1`
	_, err := r.pool.Exec(ctx, query, apiKeyID, time.Now())
	return err
}

// ListAPIKeysByTenant trả về danh sách keys của tenant (sắp xếp newest first)
func (r *APIKeyRepo) ListAPIKeysByTenant(tenantID string) ([]models.APIKeyModel, error) {
	ctx := context.Background()
	query := `SELECT id, hashed_key, tenant_id, created_at, expires_at, is_active, updated_at, coalesce(secret_shown,false)
			  FROM keys_service.api_keys WHERE tenant_id = $1 ORDER BY created_at DESC`
	rows, err := r.pool.Query(ctx, query, tenantID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []models.APIKeyModel
	for rows.Next() {
		var k models.APIKeyModel
		err := rows.Scan(&k.ID, &k.HashedKey, &k.TenantID, &k.CreatedAt, &k.ExpiresAt, &k.IsActive, &k.UpdatedAt, &k.SecretShown)
		if err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, nil
}

// UpdateAPIKeySecret thay đổi giá trị key (rotate) và trả về model đã update
// newHashedKey is the bcrypt/hashed secret
func (r *APIKeyRepo) UpdateAPIKeySecret(apiKeyID string, newHashedKey string) (models.APIKeyModel, error) {
	ctx := context.Background()
	query := `UPDATE keys_service.api_keys SET hashed_key = $2, updated_at = $3, secret_shown = FALSE WHERE id = $1 RETURNING id, hashed_key, tenant_id, created_at, expires_at, is_active, updated_at, coalesce(secret_shown,false)`
	row := r.pool.QueryRow(ctx, query, apiKeyID, newHashedKey, time.Now())

	var apiKey models.APIKeyModel
	err := row.Scan(&apiKey.ID, &apiKey.HashedKey, &apiKey.TenantID, &apiKey.CreatedAt, &apiKey.ExpiresAt, &apiKey.IsActive, &apiKey.UpdatedAt, &apiKey.SecretShown)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return apiKey, nil
}

// MarkSecretShown sets secret_shown = true for a key
func (r *APIKeyRepo) MarkSecretShown(apiKeyID string) error {
	ctx := context.Background()
	query := `UPDATE keys_service.api_keys SET secret_shown = TRUE, updated_at = $2 WHERE id = $1`
	_, err := r.pool.Exec(ctx, query, apiKeyID, time.Now())
	return err
}
