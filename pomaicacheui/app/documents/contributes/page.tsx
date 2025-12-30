"use client";

import React from "react";

export default function ContributePage() {
    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-12 border-b border-gray-200 dark:border-gray-700 pb-8 text-center">
                <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white tracking-tight">
                    Contributing to Pomai Cache
                </h1>
                <p className="mt-4 text-lg text-gray-600 dark:text-gray-300">
                    We're building the fastest discovery-based caching layer, and we'd love your help to make it even better.
                </p>
                <div className="mt-6 flex justify-center gap-4">
                    <a
                        href="https://github.com/AutoCookies/pomai-cache"
                        target="_blank"
                        className="px-6 py-2 bg-gray-900 dark:bg-white dark:text-gray-900 text-white rounded-full font-medium hover:opacity-90 transition-opacity flex items-center"
                    >
                        View on GitHub
                    </a>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
                <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm">
                    <h3 className="text-xl font-bold mb-3 text-indigo-600">🐛 Report Bugs</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                        Found an issue with server discovery or the SDK? Open an issue on GitHub with a clear reproduction case. We aim to fix critical bugs within 48 hours.
                    </p>
                </div>
                <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm">
                    <h3 className="text-xl font-bold mb-3 text-teal-600">🚀 Suggest Features</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                        Want support for Python, Rust, or a new cache eviction policy? Start a discussion in the GitHub repo to share your ideas with the community.
                    </p>
                </div>
            </div>

            <section className="space-y-8">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Our Workflow</h2>

                <div className="space-y-6">
                    <div className="flex gap-4">
                        <div className="flex-none w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 flex items-center justify-center font-bold">1</div>
                        <div>
                            <h4 className="font-bold">Fork & Clone</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Fork the repository to your own account and clone it locally to start working on your changes.</p>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <div className="flex-none w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 flex items-center justify-center font-bold">2</div>
                        <div>
                            <h4 className="font-bold">Branching</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Create a feature branch (e.g., <code>feat/new-language-sdk</code> or <code>fix/discovery-timeout</code>).</p>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <div className="flex-none w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 flex items-center justify-center font-bold">3</div>
                        <div>
                            <h4 className="font-bold">Test Your Changes</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                                For JS/TS, run the local example using <code>npm link</code>. For Go, ensure all tests in <code>client_test.go</code> pass.
                            </p>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <div className="flex-none w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 flex items-center justify-center font-bold">4</div>
                        <div>
                            <h4 className="font-bold">Submit a Pull Request</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Push your branch and open a PR. Please include a clear description of what changed and why.</p>
                        </div>
                    </div>
                </div>
            </section>

            <section className="mt-16 p-8 bg-gray-50 dark:bg-gray-900 rounded-3xl border border-gray-200 dark:border-gray-800">
                <h3 className="text-xl font-bold mb-4">Core Principles</h3>
                <ul className="space-y-4 text-sm text-gray-600 dark:text-gray-400">
                    <li className="flex items-start">
                        <span className="text-indigo-500 mr-3">✔</span>
                        <span><strong>Zero-Config First:</strong> Every SDK must prioritize ease of use, ensuring users only need their API Key to get started.</span>
                    </li>
                    <li className="flex items-start">
                        <span className="text-indigo-500 mr-3">✔</span>
                        <span><strong>Performance:</strong> Minimize discovery overhead by implementing local caching of server nodes.</span>
                    </li>
                    <li className="flex items-start">
                        <span className="text-indigo-500 mr-3">✔</span>
                        <span><strong>Security:</strong> Never hardcode sensitive production URLs or keys into example files.</span>
                    </li>
                </ul>
            </section>

            <footer className="mt-20 text-center">
                <p className="text-gray-500">
                    Questions? Reach out to us on <a href="https://github.com/AutoCookies/pomai-cache/discussions" className="text-indigo-600 font-medium">GitHub Discussions</a>.
                </p>
            </footer>
        </div>
    );
}