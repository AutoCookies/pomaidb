package persistence

import (
	"context"
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

// CreateAPIKey thêm API Key vào cơ sở dữ liệu
func (r *APIKeyRepo) CreateAPIKey(apiKey models.APIKeyModel) error {
	ctx := context.Background()
	query := `
		INSERT INTO keys_service.api_keys (id, key, tenant_id, created_at, expires_at, is_active) 
		VALUES ($1, $2, $3, $4, $5, $6)`
	_, err := r.pool.Exec(ctx, query, apiKey.ID, apiKey.Key, apiKey.TenantID, apiKey.CreatedAt, apiKey.ExpiresAt, apiKey.IsActive)
	if err != nil {
		return err
	}
	return nil
}

// GetAPIKeyByID tìm API Key bằng ID
func (r *APIKeyRepo) GetAPIKeyByID(apiKeyID string) (models.APIKeyModel, error) {
	ctx := context.Background()
	query := `SELECT id, key, tenant_id, created_at, expires_at, is_active 
			  FROM keys_service.api_keys WHERE id = $1`
	row := r.pool.QueryRow(ctx, query, apiKeyID)

	var apiKey models.APIKeyModel
	err := row.Scan(&apiKey.ID, &apiKey.Key, &apiKey.TenantID, &apiKey.CreatedAt, &apiKey.ExpiresAt, &apiKey.IsActive)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return apiKey, nil
}

// GetAPIKeyByKey tìm API Key với chính Key
func (r *APIKeyRepo) GetAPIKeyByKey(key string) (models.APIKeyModel, error) {
	ctx := context.Background()
	query := `SELECT id, key, tenant_id, created_at, expires_at, is_active 
			  FROM keys_service.api_keys WHERE key = $1`
	row := r.pool.QueryRow(ctx, query, key)

	var apiKey models.APIKeyModel
	err := row.Scan(&apiKey.ID, &apiKey.Key, &apiKey.TenantID, &apiKey.CreatedAt, &apiKey.ExpiresAt, &apiKey.IsActive)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return apiKey, nil
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
	query := `SELECT id, key, tenant_id, created_at, expires_at, is_active, updated_at 
			  FROM keys_service.api_keys WHERE tenant_id = $1 ORDER BY created_at DESC`
	rows, err := r.pool.Query(ctx, query, tenantID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []models.APIKeyModel
	for rows.Next() {
		var k models.APIKeyModel
		// Ensure your models.APIKeyModel has UpdatedAt if scanning
		err := rows.Scan(&k.ID, &k.Key, &k.TenantID, &k.CreatedAt, &k.ExpiresAt, &k.IsActive, &k.UpdatedAt)
		if err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, nil
}

// UpdateAPIKeySecret thay đổi giá trị key (rotate) và trả về model đã update
func (r *APIKeyRepo) UpdateAPIKeySecret(apiKeyID string, newKey string) (models.APIKeyModel, error) {
	ctx := context.Background()
	query := `UPDATE keys_service.api_keys SET key = $2, updated_at = $3 WHERE id = $1 RETURNING id, key, tenant_id, created_at, expires_at, is_active, updated_at`
	row := r.pool.QueryRow(ctx, query, apiKeyID, newKey, time.Now())

	var apiKey models.APIKeyModel
	err := row.Scan(&apiKey.ID, &apiKey.Key, &apiKey.TenantID, &apiKey.CreatedAt, &apiKey.ExpiresAt, &apiKey.IsActive, &apiKey.UpdatedAt)
	if err != nil {
		return models.APIKeyModel{}, err
	}
	return apiKey, nil
}
