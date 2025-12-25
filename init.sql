-- File: init.sql
CREATE SCHEMA IF NOT EXISTS keys_service;

CREATE TABLE IF NOT EXISTS keys_service.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashed_key TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    secret_shown BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON keys_service.api_keys(tenant_id);