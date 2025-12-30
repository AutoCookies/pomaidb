"use client";

import React from "react";

const setExample = `// Set a string value (Node / Browser)
import { PomaiClient } from '@autocookie/pomai-cache';

const client = new PomaiClient(process.env.POMAI_CACHE_KEY || 'YOUR_API_KEY');

await client.set('greeting', 'Hello Pomai!', { ttl: 60 }); // ttl in seconds
`;

const getExample = `// Get a string value
const val = await client.get('greeting');
if (val === null) {
  console.log('Cache miss');
} else {
  console.log('Value:', val);
}
`;

const deleteExample = `// Delete a key
await client.delete('greeting'); // no error if missing
`;

const jsonExample = `// Store and retrieve JSON
await client.set('user:101', { id: 101, name: 'Alice' }, { ttl: 300 });

const user = await client.getJSON<{ id: number; name: string }>('user:101');
console.log(user?.name);
`;

const counterExample = `// Atomic counters (incr/decr)
const v1 = await client.incr('page:views:home');      // +1
const v2 = await client.incr('page:views:home', 10);  // +10
const v3 = await client.decr('page:views:home', 3);   // -3
console.log(v1, v2, v3);
`;

const ttlExample = `// Check TTL (remaining seconds)
const remaining = await client.ttl('user:101'); // -2 = missing, -1 = no TTL set
console.log('TTL remaining:', remaining);
`;

export default function OperationsPage() {
    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Interacting with the Cache</h1>
                <p className="mt-3 text-gray-600 dark:text-gray-300">
                    This guide breaks down common cache interactions into separate sections: basic string operations, JSON binding, atomic counters, TTL checks, and direct HTTP examples.
                </p>
            </header>

            <section className="space-y-6">
                {/* Basic Set */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Set (PUT) — Store a value</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Use <code>client.set(key, value, {`{ ttl?: number }`})</code> to store string or JSON-serializable values. TTL is in seconds; omit or set 0 for no expiration.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{setExample}</code>
                    </pre>
                    <ul className="mt-3 text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                        <li>When passing an object, the SDK serializes it as JSON and sets Content-Type to application/json.</li>
                        <li>On HTTP level this maps to PUT /cache/:key with optional query ?ttl=&lt;seconds&gt;.</li>
                    </ul>
                </div>

                {/* Get */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Get (GET) — Retrieve a value</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Use <code>client.get(key)</code> to retrieve a string/bytes payload. Returns <code>null</code> on miss.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{getExample}</code>
                    </pre>
                    <ul className="mt-3 text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                        <li>HTTP: GET /cache/:key — 200 with body on hit, 404 on miss.</li>
                        <li>Use <code>getJSON</code> if you stored JSON and want automatic parsing.</li>
                    </ul>
                </div>

                {/* Delete */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Delete (DELETE) — Remove a key</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Use <code>client.delete(key)</code> to remove a key. This is idempotent — no error if the key does not exist.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{deleteExample}</code>
                    </pre>
                </div>

                {/* JSON */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">JSON — Store and retrieve structured data</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        The SDK convenience methods keep JSON handling simple: pass an object to <code>set</code> and use <code>getJSON</code> to receive parsed objects.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{jsonExample}</code>
                    </pre>
                    <ul className="mt-3 text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                        <li>If JSON parsing fails, <code>getJSON</code> throws an error to help you detect corrupted or unexpected payloads.</li>
                        <li>JSON values are stored with Content-Type: application/json.</li>
                    </ul>
                </div>

                {/* Counters */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Atomic Counters — incr / decr</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Use <code>incr(key, delta?)</code> and <code>decr(key, delta?)</code> to perform atomic numeric updates. These calls return the new numeric value.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{counterExample}</code>
                    </pre>
                    <ul className="mt-3 text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                        <li>On the HTTP level this maps to POST /cache/:key/incr?delta=&lt;n&gt;.</li>
                        <li>Ensure the stored value is numeric (or the server will return an error), or initialize it to "0" before increments.</li>
                    </ul>
                </div>

                {/* TTL */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">TTL / HEAD — Check remaining time</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Use <code>client.ttl(key)</code> to check remaining TTL (in seconds). Return values:
                    </p>
                    <ul className="mt-3 text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                        <li><code>-2</code> — key does not exist</li>
                        <li><code>-1</code> — key exists but has no TTL</li>
                        <li>non-negative number — seconds remaining</li>
                    </ul>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{ttlExample}</code>
                    </pre>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                        At HTTP level, HEAD /cache/:key returns headers, including <code>X-Cache-TTL-Remaining</code> when a TTL is set.
                    </p>
                </div>

                {/* Error handling */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Error handling & best practices</h2>
                    <ul className="text-sm list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2">
                        <li>Most SDK calls throw an error when the HTTP response is not ok — wrap calls in try/catch.</li>
                        <li>Use short TTLs for ephemeral data. For counters or persistent session data, choose TTL accordingly.</li>
                        <li>To avoid unnecessary discovery calls, reuse the client instance across your application lifetime.</li>
                        <li>Keep API keys secret and rotate them if compromised.</li>
                    </ul>
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                    <p>
                        Need these broken down into separate pages (one file per operation) or want runnable playground examples? I can split each section into its own route/page with interactive code samples.
                    </p>
                </div>
            </section>
        </div>
    );
}