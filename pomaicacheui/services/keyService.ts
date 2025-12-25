import http from "@/lib/http";
import { KEY_ENDPOINTS } from "@/lib/key/keyApi";

export interface KeyPublic {
    id: string;
    tenantId?: string;
    createdAt?: string;
    updatedAt?: string;
    expiresAt?: string;
    isActive?: boolean;
    secretShown?: boolean;
    name?: string;
    description?: string;
}

export interface CreateRotateResp {
    key_id?: string;
    secret?: string;
    tenant_id?: string;
    expires_at?: string;
    secret_shown?: boolean;
    message?: string;
    key?: {
        id?: string;
        secret?: string;
        keyId?: string;
        key?: string;
    };
}

export interface ListResp {
    keys: KeyPublic[];
    message?: string;
}

export interface DeleteResp {
    success: boolean;
    message?: string;
}

export interface ValidateResp {
    valid: boolean;
    tenantId?: string;
    message?: string;
}

interface CreateKeyParams {
    name: string;
    description?: string;
}

interface RotateKeyParams {
    keyId: string;
}

interface ValidateKeyParams {
    key: string;
}

const getString = (v: any): string | undefined => {
    if (v === null || typeof v === "undefined") return undefined;
    return String(v);
};

export const createKey = async (data: CreateKeyParams): Promise<CreateRotateResp> => {
    const res = (await http.post(KEY_ENDPOINTS.CREATE, data)) as any;

    const keyId =
        getString(res.key_id) ||
        getString(res.id) ||
        getString(res.key?.id) ||
        getString(res.key?.keyId);

    const secret =
        getString(res.secret) ||
        getString(res.key?.secret) ||
        getString(res.key?.key);

    const tenant_id = getString(res.tenant_id) || getString(res.tenantId);
    const expires_at = getString(res.expires_at) || getString(res.expiresAt);

    // secret_shown: prefer explicit presence of `secret_shown`, fallback to `secretShown` if present
    let secret_shown: boolean | undefined = undefined;
    if ("secret_shown" in res) {
        secret_shown = Boolean(res.secret_shown);
    } else if ("secretShown" in res) {
        secret_shown = Boolean((res as any).secretShown);
    }

    const out: CreateRotateResp = {
        key_id: keyId,
        secret,
        tenant_id,
        expires_at,
        secret_shown,
        message: getString(res.message),
        key: res.key ? { ...res.key } : undefined,
    };

    return out;
};

export const listKeys = async (): Promise<ListResp> => {
    const res = (await http.get(KEY_ENDPOINTS.LIST)) as any;

    const rawKeys: any[] = Array.isArray(res.keys)
        ? res.keys
        : Array.isArray(res.data)
            ? res.data
            : [];

    const keys: KeyPublic[] = rawKeys.map((k: any) => ({
        id: getString(k.id) || getString(k.ID) || getString(k.keyId) || "",
        tenantId: getString(k.tenant_id) || getString(k.tenantId),
        createdAt: getString(k.created_at) || getString(k.createdAt) || getString(k.created),
        updatedAt: getString(k.updated_at) || getString(k.updatedAt) || getString(k.updated),
        expiresAt: getString(k.expires_at) || getString(k.expiresAt),
        isActive: typeof k.is_active !== "undefined" ? Boolean(k.is_active) : typeof k.isActive !== "undefined" ? Boolean(k.isActive) : undefined,
        secretShown: typeof k.secret_shown !== "undefined" ? Boolean(k.secret_shown) : typeof k.secretShown !== "undefined" ? Boolean(k.secretShown) : undefined,
        name: getString(k.name) || getString(k.label),
        description: getString(k.description) || getString(k.desc),
    }));

    return { keys, message: getString(res.message) };
};

export const rotateKey = async (data: RotateKeyParams): Promise<CreateRotateResp> => {
    const res = (await http.post(KEY_ENDPOINTS.ROTATE, data)) as any;

    const keyId =
        getString(res.key_id) ||
        getString(res.id) ||
        getString(res.key?.id);

    const secret =
        getString(res.secret) ||
        getString(res.key?.secret) ||
        getString(res.key?.key);

    const tenant_id = getString(res.tenant_id) || getString(res.tenantId);
    const expires_at = getString(res.expires_at) || getString(res.expiresAt);

    let secret_shown: boolean | undefined = undefined;
    if ("secret_shown" in res) {
        secret_shown = Boolean(res.secret_shown);
    } else if ("secretShown" in res) {
        secret_shown = Boolean((res as any).secretShown);
    }

    const out: CreateRotateResp = {
        key_id: keyId,
        secret,
        tenant_id,
        expires_at,
        secret_shown,
        message: getString(res.message),
        key: res.key ? { ...res.key } : undefined,
    };

    return out;
};

export const deleteKey = async (keyId: string): Promise<DeleteResp> => {
    const res = (await http.delete(`${KEY_ENDPOINTS.DELETE}?id=${encodeURIComponent(keyId)}`)) as any;
    return {
        success: Boolean(res.success ?? res.ok ?? false),
        message: getString(res.message),
    };
};

export const validateKey = async (data: ValidateKeyParams): Promise<ValidateResp> => {
    const res = (await http.get(`${KEY_ENDPOINTS.VALIDATE}?key=${encodeURIComponent(data.key)}`)) as any;

    const valid =
        typeof res.is_valid !== "undefined"
            ? Boolean(res.is_valid)
            : typeof res.valid !== "undefined"
                ? Boolean(res.valid)
                : Boolean(res.validated ?? false);

    const tenantId = getString(res.tenant_id) || getString(res.tenantId) || getString(res.tenant);
    const message = getString(res.message) || (valid ? "ok" : "invalid");

    return { valid, tenantId, message };
};

export default {
    createKey,
    listKeys,
    rotateKey,
    deleteKey,
    validateKey,
};