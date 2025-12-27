"use client";

import React from "react";

const nodeExample = `// Node.js (server)
import { PomaiClient } from '@autocookie/pomai-cache';

const client = new PomaiClient(process.env.POMAI_CACHE_KEY || 'YOUR_API_KEY');

async function run() {
  // Store a string with TTL 60s
  await client.set('greeting', 'Hello Pomai!', { ttl: 60 });

  // Retrieve it
  const val = await client.get('greeting');
  console.log('greeting =', val);
}

run().catch(console.error);`;

const browserExample = `// Browser (bundler / Deno / Workers)
import { PomaiClient } from '@autocookie/pomai-cache';

// Initialize with only your API key
const client = new PomaiClient('YOUR_API_KEY');

await client.set('session:user:1', { id: 1, name: 'Alice' }, { ttl: 300 });
const user = await client.getJSON('session:user:1');
console.log(user);`;

const curlExample = `# 1) (Optional) Discover base_url if you don't have it
curl -s -H "X-API-Key: YOUR_API_KEY" "https://discovery.example.com/api-key/discover"

# Response example:
# { "base_url": "https://pomaicache-api.example.com/v1" }

# 2) PUT value (once you know base_url)
curl -X PUT "https://pomaicache-api.example.com/v1/cache/greeting?ttl=60" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  --data "Hello via curl"

# 3) GET value
curl -H "X-API-Key: YOUR_API_KEY" "https://pomaicache-api.example.com/v1/cache/greeting"
`;

export default function QuickstartPage() {
    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Quickstart — Connect to Pomai Cache</h1>
                <p className="mt-3 text-gray-600 dark:text-gray-300">
                    You only need your API key to use Pomai Cache. If your environment requires a discovery URL, provide it via config or the POMAI_DISCOVERY_URL environment variable; otherwise the SDK will use the default local discovery URL.
                </p>
            </header>

            <section className="space-y-6">
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">1. Install</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Install the JS/TS package (example package name: <code>@autocookie/pomai-cache</code>).
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>npm install @autocookie/pomai-cache</code>
                    </pre>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">2. Configure your API key</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Store your API key in an environment variable (or provide it directly when initializing the SDK in a secure environment).
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>POMAI_CACHE_KEY=your_full_api_key_here</code>
                    </pre>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">3. Quick example — Node.js</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Initialize the client with your API key. If you need to override discovery behavior, pass <code>{`{ discoveryUrl: 'https://...' }`}</code> in the constructor or set <code>POMAI_DISCOVERY_URL</code>.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{nodeExample}</code>
                    </pre>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">4. Example — Browser / Edge environments</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        The same SDK works in browsers, Workers, Deno, and other JS runtimes. Initialize with your API key.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{browserExample}</code>
                    </pre>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <h2 className="text-xl font-semibold mb-3">5. Example — curl (HTTP)</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        To call the HTTP API directly, you can optionally query the discovery endpoint to obtain the base_url, then perform PUT/GET requests.
                    </p>
                    <pre className="mt-3 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{curlExample}</code>
                    </pre>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm text-sm text-gray-600 dark:text-gray-300">
                    <h3 className="font-semibold mb-2">Important notes</h3>
                    <ul className="list-disc list-inside space-y-1">
                        <li>You only need to provide your API key to initialize the SDK in most setups.</li>
                        <li>If your deployment requires a specific discovery URL, set the <code>POMAI_DISCOVERY_URL</code> environment variable or pass <code>discoveryUrl</code> in the client config.</li>
                        <li>The SDK can cache a discovered base_url in localStorage (when available) to reduce repeated discovery calls.</li>
                        <li>Keep your API key secret — do not commit it to public repositories. Use environment variables or a secret manager.</li>
                    </ul>
                </div>

                <div className="text-sm text-gray-500 dark:text-gray-400">
                    <p>
                        Want examples for Go or more advanced patterns (TTL management, counters)? I can add a "Next steps" section using your provided main.go sample.
                    </p>
                </div>
            </section>
        </div>
    );
}