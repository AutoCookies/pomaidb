"use client";

import React from "react";

export default function IntroPage() {
    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Pomai Cache</h1>
                <p className="mt-3 text-gray-600 dark:text-gray-300">
                    Pomai Cache is a lightweight, secure HTTP cache service that gives applications
                    a simple, fast way to store and retrieve ephemeral data. It provides a
                    Redis-like developer experience over standard HTTP endpoints and API key authentication.
                </p>
            </header>

            <section className="space-y-6">
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">What it does</h2>
                    <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Pomai Cache exposes simple endpoints (PUT/GET/DELETE/HEAD) to store arbitrary
                        byte payloads with optional TTL. It's designed for low-latency caching, temporary
                        storage, session/state sharing, and fast lookups across distributed services.
                    </p>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Key benefits</h2>
                    <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 text-sm space-y-2">
                        <li>Simple HTTP API — easy to call from any language or platform.</li>
                        <li>API key authentication — keys are shown once and can be rotated or revoked.</li>
                        <li>Per-tenant routing — secure multi-tenant isolation via a gateway.</li>
                        <li>Small SDKs available (Go, JS/TS, Python) to simplify integration.</li>
                    </ul>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">Security & best practices</h2>
                    <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 text-sm space-y-2">
                        <li>Do not embed API keys in public repositories — use environment variables or secret managers.</li>
                        <li>Rotate keys periodically and revoke any compromised keys immediately.</li>
                        <li>Clients talk to a public gateway; internal topology and server addresses are never exposed.</li>
                    </ul>
                </div>

                <div className="text-sm text-gray-500 dark:text-gray-400">
                    <p>
                        To get started, create an API key in the dashboard and use one of our SDKs or the HTTP API
                        to store and retrieve cached values.
                    </p>
                </div>
            </section>
        </div>
    );
}