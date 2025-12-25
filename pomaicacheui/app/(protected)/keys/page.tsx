"use client";

import React, { useCallback, useEffect, useState } from "react";
import keyService, { KeyPublic, CreateRotateResp } from "@/services/keyService";
import { formatDbTimestamp } from "@/utils/dateUtils";
import "remixicon/fonts/remixicon.css";

type ListKey = KeyPublic & {
    // frontend won't persist secret in list
    secret?: never;
    name?: string;
    description?: string;
};

export default function ApiKeyPage() {
    const [keys, setKeys] = useState<ListKey[]>([]);
    const [loading, setLoading] = useState(false);
    const [opLoading, setOpLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Create states
    const [showCreate, setShowCreate] = useState(false);
    const [createName, setCreateName] = useState("");
    const [createDescription, setCreateDescription] = useState("");
    const [createdSecret, setCreatedSecret] = useState<string | null>(null);
    const [createdKeyId, setCreatedKeyId] = useState<string | null>(null);

    // Validate
    const [validateInput, setValidateInput] = useState("");
    const [validateResult, setValidateResult] = useState<{ valid: boolean; msg: string } | null>(null);

    const fetchKeys = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await keyService.listKeys();
            const raw = (res?.keys ?? []) as any[];
            const mapped: ListKey[] = raw.map((k) => ({
                id: k.id ?? k.ID ?? k.keyId ?? "",
                tenantId: k.tenant_id ?? k.tenantId ?? undefined,
                createdAt: k.created_at ?? k.createdAt ?? k.created ?? undefined,
                updatedAt: k.updated_at ?? k.updatedAt ?? k.updated ?? undefined,
                expiresAt: k.expires_at ?? k.expiresAt ?? undefined,
                isActive: typeof k.is_active !== "undefined" ? Boolean(k.is_active) : typeof k.isActive !== "undefined" ? Boolean(k.isActive) : undefined,
                secretShown: typeof k.secret_shown !== "undefined" ? Boolean(k.secret_shown) : typeof k.secretShown !== "undefined" ? Boolean(k.secretShown) : undefined,
                name: k.name ?? k.label ?? undefined,
                description: k.description ?? k.desc ?? undefined,
            }));
            setKeys(mapped);
        } catch (err: any) {
            setError(err?.message || "Failed to load keys");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchKeys();
    }, [fetchKeys]);

    const handleCreate = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!createName.trim()) return;
        setOpLoading("create");
        setError(null);
        setCreatedSecret(null);
        setCreatedKeyId(null);

        try {
            const res = (await keyService.createKey({
                name: createName.trim(),
                description: createDescription.trim() || undefined,
            })) as CreateRotateResp;

            // Normalize secret and key id (createKey returns normalized shape)
            const keyId = res.key_id ?? (res.key && res.key.id) ?? undefined;
            const secret = res.secret ?? (res.key && (res.key.secret ?? (res.key as any).key)) ?? undefined;

            // Add new item to list (metadata only)
            const newItem: ListKey = {
                id: keyId ?? `temp-${Date.now()}`,
                name: createName.trim(),
                description: createDescription.trim() || undefined,
                createdAt: new Date().toISOString(),
                // do not store secret in list
            };
            setKeys((prev) => [newItem, ...prev]);

            if (secret) {
                setCreatedSecret(secret);
                setCreatedKeyId(keyId ?? null);
                // try to copy automatically, ignore errors
                try {
                    await navigator.clipboard.writeText(secret);
                } catch {
                    // ignore
                }
            }

            // reset create form
            setShowCreate(false);
            setCreateName("");
            setCreateDescription("");
        } catch (err: any) {
            setError(err?.message || "Failed to create key");
        } finally {
            setOpLoading(null);
        }
    };

    const handleRotate = async (keyId: string) => {
        if (!confirm("Rotate this key? The old secret will stop working immediately.")) return;
        setOpLoading(keyId);
        setError(null);
        try {
            const res = (await keyService.rotateKey({ keyId })) as CreateRotateResp;
            const secret = res.secret ?? (res.key && (res.key.secret ?? (res.key as any).key)) ?? undefined;

            // update updatedAt
            setKeys((prev) => prev.map((k) => (k.id === keyId ? { ...k, updatedAt: new Date().toISOString() } : k)));

            if (secret) {
                setCreatedSecret(secret);
                setCreatedKeyId(keyId);
                try {
                    await navigator.clipboard.writeText(secret);
                } catch {
                    // ignore
                }
                // notify user
                alert("New secret generated — it is shown once in the banner. Please copy it now.");
            }
        } catch (err: any) {
            alert(err?.message || "Failed to rotate key");
        } finally {
            setOpLoading(null);
        }
    };

    const handleDelete = async (keyId: string) => {
        if (!confirm("Delete this key? This will deactivate it.")) return;
        setOpLoading(keyId);
        setError(null);
        try {
            await keyService.deleteKey(keyId);
            setKeys((prev) => prev.filter((k) => k.id !== keyId));
        } catch (err: any) {
            alert(err?.message || "Failed to delete key");
        } finally {
            setOpLoading(null);
        }
    };

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text).catch(() => {
            // ignore
        });
    };

    const handleValidate = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!validateInput.trim()) return;
        setOpLoading("validate");
        setValidateResult(null);
        try {
            const res = await keyService.validateKey({ key: validateInput.trim() }); const valid = !!res.valid;
            setValidateResult({ valid, msg: valid ? "Valid API Key" : "Invalid API Key" });
        } catch (err: any) {
            setValidateResult({ valid: false, msg: err?.message || "Validation error" });
        } finally {
            setOpLoading(null);
        }
    };

    const acknowledgeSecret = () => {
        // user confirmed they copied the one-time secret; hide banner
        setCreatedSecret(null);
        setCreatedKeyId(null);
        // optional: call ack endpoint if implemented
        // if (createdKeyId) keyService.ackSecretShown({ keyId: createdKeyId }).catch(()=>{/*ignore*/});
    };

    return (
        <div className="max-w-5xl mx-auto py-8 px-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">API Keys</h1>
                    <p className="text-sm text-gray-500 mt-1">Manage authentication keys for your applications.</p>
                </div>
                <button
                    onClick={() => setShowCreate(true)}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg text-sm font-medium hover:opacity-90 transition-opacity shadow-sm"
                >
                    <i className="ri-add-line text-lg" /> Create New Key
                </button>
            </div>

            {error && (
                <div className="mb-6 p-3 bg-red-50 text-red-600 text-sm rounded-lg border border-red-100">{error}</div>
            )}

            {/* One-time secret banner */}
            {createdSecret && (
                <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-bold text-green-800 dark:text-green-300">Your API Key (shown once)</span>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => handleCopy(createdSecret)}
                                className="text-xs bg-white dark:bg-black px-2 py-1 rounded border shadow-sm hover:bg-gray-50"
                            >
                                Copy
                            </button>
                            <button onClick={acknowledgeSecret} className="text-xs px-2 py-1 rounded border shadow-sm hover:bg-gray-50">
                                I copied it
                            </button>
                        </div>
                    </div>
                    <div className="font-mono text-xs text-green-900 dark:text-green-100 break-all bg-white/50 dark:bg-black/20 p-2 rounded">
                        {createdSecret}
                    </div>
                    <p className="text-xs text-green-700 mt-2">This secret is shown only once. Store it securely — it will not be visible again.</p>
                </div>
            )}

            {/* List */}
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl shadow-sm overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-gray-50 dark:bg-gray-800/50 border-b border-gray-100 dark:border-gray-800 text-gray-500 font-medium">
                            <tr>
                                <th className="px-6 py-4 w-[35%]">Name & Identity</th>
                                <th className="px-6 py-4 w-[40%]">Secret</th>
                                <th className="px-6 py-4 w-[15%]">Created</th>
                                <th className="px-6 py-4 w-[10%] text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                            {loading ? (
                                <tr>
                                    <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                                        Loading...
                                    </td>
                                </tr>
                            ) : keys.length === 0 ? (
                                <tr>
                                    <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                                        No keys found.
                                    </td>
                                </tr>
                            ) : (
                                keys.map((k) => (
                                    <tr key={k.id} className="group hover:bg-gray-50/50 dark:hover:bg-gray-800/30 transition-colors">
                                        <td className="px-6 py-4 align-top">
                                            <div className="font-semibold text-gray-900 dark:text-white">{k.name ?? k.id}</div>
                                            {k.description && (
                                                <div className="text-xs text-gray-500 truncate max-w-[200px]" title={k.description}>
                                                    {k.description}
                                                </div>
                                            )}
                                            <div className="mt-1 font-mono text-[10px] text-gray-400 select-all">ID: {k.id}</div>
                                        </td>

                                        <td className="px-6 py-4 align-middle">
                                            <span className="text-xs text-gray-400 italic">Secret hidden (Rotate to regenerate)</span>
                                        </td>

                                        <td className="px-6 py-4 align-middle text-gray-500 text-xs">{formatDbTimestamp(k.createdAt, k.createdAt)}</td>

                                        <td className="px-6 py-4 align-middle text-right">
                                            <div className="flex items-center justify-end gap-1">
                                                <button
                                                    onClick={() => handleRotate(k.id)}
                                                    disabled={!!opLoading}
                                                    className="p-2 rounded text-gray-400 hover:text-amber-600 hover:bg-amber-50 transition-all"
                                                    title="Rotate Key"
                                                >
                                                    <i className={`ri-refresh-line ${opLoading === k.id ? "animate-spin text-amber-600" : ""}`} />
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(k.id)}
                                                    disabled={!!opLoading}
                                                    className="p-2 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 transition-all"
                                                    title="Delete Key"
                                                >
                                                    <i className="ri-delete-bin-line" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Validate */}
            <div className="mt-8 max-w-2xl">
                <details className="group text-sm text-gray-500">
                    <summary className="cursor-pointer hover:text-gray-700 font-medium select-none list-none flex items-center gap-2">
                        <i className="ri-shield-check-line" /> Validate an API Key
                        <i className="ri-arrow-down-s-line group-open:rotate-180 transition-transform" />
                    </summary>
                    <div className="mt-3 pl-6 border-l-2 border-gray-100">
                        <form onSubmit={handleValidate} className="flex gap-2 items-center">
                            <input
                                value={validateInput}
                                onChange={(e) => {
                                    setValidateInput(e.target.value);
                                    setValidateResult(null);
                                }}
                                className="flex-1 px-3 py-2 rounded border border-gray-200 dark:border-gray-800 bg-transparent text-sm"
                                placeholder="Paste key to check validity..."
                            />
                            <button disabled={opLoading === "validate"} type="submit" className="px-4 py-2 bg-gray-900 text-white rounded hover:bg-gray-800 text-sm">
                                Check
                            </button>
                        </form>
                        {validateResult && (
                            <div className={`mt-2 text-xs font-medium ${validateResult.valid ? "text-green-600" : "text-red-600"}`}>
                                {validateResult.valid ? <i className="ri-checkbox-circle-fill mr-1" /> : <i className="ri-close-circle-fill mr-1" />}
                                {validateResult.msg}
                            </div>
                        )}
                    </div>
                </details>
            </div>

            {/* Create modal */}
            {showCreate && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setShowCreate(false)} />
                    <div className="relative bg-white dark:bg-gray-900 rounded-xl shadow-2xl w-full max-w-md p-6 animate-in fade-in zoom-in-95 duration-200">
                        <h3 className="text-lg font-bold mb-4">Create New API Key</h3>
                        <form onSubmit={handleCreate} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
                                <input
                                    autoFocus
                                    value={createName}
                                    onChange={(e) => setCreateName(e.target.value)}
                                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 text-sm focus:ring-2 focus:ring-black focus:border-transparent outline-none"
                                    placeholder="e.g. Production Service"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">Description</label>
                                <textarea
                                    value={createDescription}
                                    onChange={(e) => setCreateDescription(e.target.value)}
                                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 text-sm focus:ring-2 focus:ring-black outline-none h-20 resize-none"
                                    placeholder="Optional description..."
                                />
                            </div>
                            <div className="flex justify-end gap-2 mt-6">
                                <button type="button" onClick={() => setShowCreate(false)} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">
                                    Cancel
                                </button>
                                <button type="submit" disabled={!createName.trim() || opLoading === "create"} className="px-4 py-2 text-sm bg-black text-white rounded-lg hover:bg-gray-800 disabled:opacity-50">
                                    {opLoading === "create" ? "Creating..." : "Create Key"}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}