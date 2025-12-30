"use client";

import React from "react";

const nodeExample = `// Node.js (Server)
import { PomaiClient } from '@autocookie/pomai-cache';

// The SDK automatically handles server discovery using your API Key
const client = new PomaiClient(process.env.POMAI_CACHE_KEY || 'YOUR_API_KEY');

async function run() {
  // Store a string with a 60-second TTL
  await client.set('greeting', 'Hello Pomai!', { ttl: 60 });

  // Retrieve the value
  const val = await client.get('greeting');
  console.log('Value:', val);
}

run().catch(console.error);`;

const goExample = `// Go (Golang)
import (
    "context"
    "fmt"
    "github.com/AutoCookies/pomai-sdk"
)

func main() {
    ctx := context.Background()
    
    // Initialize the client with your API key
    client, _ := sdk.New("YOUR_API_KEY")

    // Store a JSON object
    user := User{ID: 101, Name: "AutoCookies"}
    client.SetJSON(ctx, "user:101", user, 300)

    // Atomic increment
    newCount, _ := client.Incr(ctx, "page_views", 1)
    fmt.Printf("Total Views: %d\\n", newCount)
}`;

export default function QuickstartPage() {
    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-12 border-b border-gray-200 dark:border-gray-700 pb-8">
                <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white tracking-tight">
                    Quickstart Guide
                </h1>
                <p className="mt-4 text-lg text-gray-600 dark:text-gray-300">
                    Connect to Pomai Cache in minutes. Our SDKs handle the complex routing and discovery logic for you automatically.
                </p>
            </header>

            <div className="grid grid-cols-1 gap-12">
                {/* Step 1: Install */}
                <section>
                    <h2 className="text-2xl font-bold mb-4 flex items-center">
                        <span className="bg-indigo-600 text-white rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm font-mono">1</span>
                        Install SDK
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                        <div className="p-5 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
                            <h3 className="font-bold text-sm uppercase tracking-wider text-gray-500 mb-3">JavaScript / TypeScript</h3>
                            <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm font-mono text-indigo-600 dark:text-indigo-400">
                                <code>npm install @autocookie/pomai-cache</code>
                            </pre>
                        </div>
                        <div className="p-5 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
                            <h3 className="font-bold text-sm uppercase tracking-wider text-gray-500 mb-3">Go (Golang)</h3>
                            <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm font-mono text-teal-600 dark:text-teal-400">
                                <code>go get github.com/AutoCookies/pomai-sdk</code>
                            </pre>
                        </div>
                    </div>
                </section>

                {/* Step 2: API Key */}
                <section>
                    <h2 className="text-2xl font-bold mb-4 flex items-center">
                        <span className="bg-indigo-600 text-white rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm font-mono">2</span>
                        Configure API Key
                    </h2>
                    <p className="text-gray-600 dark:text-gray-300 mb-4">
                        We recommend storing your API key in an environment variable for security.
                    </p>
                    <div className="relative group">
                        <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm font-mono border dark:border-gray-700">
                            <code>POMAI_CACHE_KEY=your_key_id.your_secret_key</code>
                        </pre>
                    </div>
                </section>

                {/* Step 3: Implementation */}
                <section className="space-y-10">
                    <h2 className="text-2xl font-bold mb-4 flex items-center">
                        <span className="bg-indigo-600 text-white rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm font-mono">3</span>
                        Usage Examples
                    </h2>

                    {/* Node.js Block */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <h3 className="text-lg font-semibold text-indigo-500 tracking-wide"># Node.js & TypeScript</h3>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                            <pre className="bg-gray-50 dark:bg-gray-900 p-5 text-sm leading-relaxed overflow-x-auto">
                                <code>{nodeExample}</code>
                            </pre>
                        </div>
                    </div>

                    {/* Go Block */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <h3 className="text-lg font-semibold text-teal-500 tracking-wide"># Go (Golang)</h3>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                            <pre className="bg-gray-50 dark:bg-gray-900 p-5 text-sm leading-relaxed overflow-x-auto">
                                <code>{goExample}</code>
                            </pre>
                        </div>
                    </div>
                </section>

                {/* Security and Best Practices */}
                <section className="p-8 bg-indigo-50 dark:bg-indigo-900/20 rounded-2xl border border-indigo-100 dark:border-indigo-900/30">
                    <h3 className="text-indigo-800 dark:text-indigo-300 font-bold text-lg mb-4 flex items-center">
                        Security Best Practices
                    </h3>
                    <ul className="space-y-3 text-indigo-900/80 dark:text-indigo-200/80">
                        <li className="flex items-start">
                            <span className="mr-2">•</span>
                            <span><strong>Zero-Config:</strong> You only need the API Key. The SDK automatically resolves the correct server nodes for your tenant.</span>
                        </li>
                        <li className="flex items-start">
                            <span className="mr-2">•</span>
                            <span><strong>Server-Side Only:</strong> Never expose your secret key in client-side code that is visible to users. Use environment variables or secret managers.</span>
                        </li>
                        <li className="flex items-start">
                            <span className="mr-2">•</span>
                            <span><strong>Automatic Optimization:</strong> The SDK caches server locations in <code>localStorage</code> (Browser) or <code>TempDir</code> (Go) to minimize discovery overhead.</span>
                        </li>
                    </ul>
                </section>
            </div>

            <footer className="mt-20 pt-10 border-t border-gray-200 dark:border-gray-700 text-center text-gray-500 dark:text-gray-400 text-sm">
                <p>
                    Need help? View the <a href="#" className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline">Full Documentation</a> or <a href="#" className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline">Contact Support</a>.
                </p>
            </footer>
        </div>
    );
}