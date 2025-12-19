import http from "@/lib/http";
import { KEY_ENDPOINTS } from "@/lib/key/keyApi";

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

export interface Key {
    id: string;
    name?: string;
    description?: string;
    key?: string; // secret (may or may not be returned by backend)
    createdAt: string;
    updatedAt?: string;
}

export const createKey = (data: CreateKeyParams) => {
    // backend may accept additional fields; if backend expects other shape, adjust accordingly
    return http.post(KEY_ENDPOINTS.CREATE, data) as Promise<{ key: Key; message?: string }>;
};

export const deleteKey = (keyId: string) => {
    // backend delete handler expects id (query or body). We use query param here.
    return http.delete(`${KEY_ENDPOINTS.DELETE}?id=${encodeURIComponent(keyId)}`) as Promise<{ success: boolean; message?: string }>;
};

export const listKeys = () => {
    return http.get(KEY_ENDPOINTS.LIST) as Promise<{ keys: Key[]; message?: string }>;
};

export const rotateKey = (data: RotateKeyParams) => {
    // backend expects { keyId: string } in body
    return http.post(KEY_ENDPOINTS.ROTATE, data) as Promise<{ key: Key; message?: string }>;
};

export const validateKey = async (data: ValidateKeyParams) => {
    // backend Validate handler uses GET ?key=...
    const res = await http.get(`${KEY_ENDPOINTS.VALIDATE}?key=${encodeURIComponent(data.key)}`);
    // backend might return { is_valid: bool } or { valid: bool } - normalize
    const anyRes = res as any;
    const valid = typeof anyRes.is_valid !== "undefined" ? anyRes.is_valid : (anyRes.valid ?? false);
    const message = anyRes.message ?? (valid ? "ok" : "invalid");
    return { valid, message } as { valid: boolean; message?: string };
};

export default {
    createKey,
    deleteKey,
    listKeys,
    rotateKey,
    validateKey,
};