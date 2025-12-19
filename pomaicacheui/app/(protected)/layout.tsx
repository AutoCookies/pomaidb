"use client"

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { MENU_LIST } from "@/constants/menu";
import { useAuth } from "@/hooks/useAuth";

/**
 * Responsive sidebar layout
 *
 * - Desktop: fixed sidebar + content area
 * - Mobile : topbar with hamburger -> opens an overlay drawer
 *
 * - Uses MENU_LIST from constants/menu.ts so menu can be extended later
 * - Displays user's displayName and email (avatar uses ui-avatars fallback)
 * - Clean, accessible and responsive
 */

const SidebarContent: React.FC<{ onNavigate?: () => void; pathname?: string }> = ({ onNavigate, pathname }) => {
    const { user } = useAuth();

    const avatar =
        // user may not have avatar property in User model; cast to any to try to read it safely
        (user as any)?.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(user?.displayName || user?.email || "User")}&background=random`;

    return (
        <div className="h-full flex flex-col bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 w-64">
            <div className="px-4 py-6">
                {/* User */}
                <div className="flex items-center gap-3 mb-6">
                    <img
                        src={avatar}
                        alt="avatar"
                        className="w-12 h-12 rounded-full object-cover bg-gray-100 dark:bg-gray-800"
                    />
                    <div className="min-w-0">
                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                            {user?.displayName || "No name"}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 truncate">{user?.email}</div>
                    </div>
                </div>

                {/* Divider */}
                <div className="mb-4 h-px bg-gray-100 dark:bg-gray-800" />

                {/* Menu */}
                <nav aria-label="Main navigation" className="space-y-1">
                    {MENU_LIST.map((item) => {
                        const isActive = pathname ? pathname === item.href || pathname.startsWith(item.href + "/") : false;
                        return (
                            <Link key={item.href} href={item.href} onClick={onNavigate}>
                                <div
                                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors
                    ${isActive ? "bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300" : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"}`}
                                    aria-current={isActive ? "page" : undefined}
                                >
                                    {/* MENU_LIST entries may include full remixicon class or just icon name.
                      We support both:
                      - if item.icon starts with "ri-" use as-is,
                      - otherwise prefix "ri-" to keep backwards compatibility.
                  */}
                                    <i className={`${item.icon?.startsWith("ri-") ? item.icon : `ri-${item.icon}`} ri-lg`} aria-hidden />
                                    <span className="truncate">{item.label}</span>
                                </div>
                            </Link>
                        );
                    })}
                </nav>
            </div>

            <div className="mt-auto px-4 py-4">
                {/* Footer area - small links or version */}
                <div className="text-xs text-gray-400 dark:text-gray-500">© {new Date().getFullYear()}</div>
            </div>
        </div>
    );
};

const ProtectedLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();

    return (
        <div className="min-h-screen flex bg-gray-50 dark:bg-gray-800">
            {/* Desktop sidebar */}
            <aside className="hidden md:flex md:flex-shrink-0">
                <SidebarContent pathname={pathname || "/"} />
            </aside>

            {/* Mobile topbar */}
            <div className="w-full md:hidden">
                <header className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setOpen(true)}
                            aria-label="Open menu"
                            className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
                        >
                            <i className="ri-menu-line ri-lg" />
                        </button>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">App</div>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* Optionally show avatar on topbar */}
                        {/* useAuth is used inside SidebarContent already so keep this minimal */}
                        <div className="w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800" />
                    </div>
                </header>

                {/* Mobile drawer (overlay) */}
                {open && (
                    <div className="fixed inset-0 z-40 flex">
                        {/* overlay */}
                        <div
                            className="fixed inset-0 bg-black/40 backdrop-blur-sm"
                            onClick={() => setOpen(false)}
                            aria-hidden
                        />

                        {/* drawer */}
                        <div className="relative w-72 max-w-full bg-white dark:bg-gray-900 shadow-xl">
                            <div className="h-full">
                                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800">
                                    <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">Menu</div>
                                    <button
                                        onClick={() => setOpen(false)}
                                        aria-label="Close menu"
                                        className="p-2 rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
                                    >
                                        <i className="ri-close-line ri-lg" />
                                    </button>
                                </div>

                                <div className="p-4">
                                    <SidebarContent onNavigate={() => setOpen(false)} pathname={pathname || "/"} />
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Main content */}
            <main className="flex-1 overflow-auto">
                <div className="min-h-full p-6">{children}</div>
            </main>
        </div>
    );
};

export default function ProtectedLayoutWrapper({ children }: { children: React.ReactNode }) {
    return (
        <ProtectedRoute>
            <ProtectedLayout>{children}</ProtectedLayout>
        </ProtectedRoute>
    );
}