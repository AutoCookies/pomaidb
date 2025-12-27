/* layout.tsx
   Layout cho phần /intro — hiển thị sidebar menu từ constants/docMenu.ts và vùng nội dung (children).
   Đây là server component (vẫn export metadata) và render menu đệ quy.
*/

import React from "react"
import Link from "next/link"
import { docMenu, DocMenuItem } from "@/constants/docsMenu"

export const metadata = {
    title: "Intro - Pomaicache",
    description: "Introduction to Pomaicache, a lightweight and secure HTTP cache service.",
}

function renderMenuItems(items: DocMenuItem[]) {
    return (
        <ul className="space-y-1">
            {items.map((item) => (
                <li key={item.id}>
                    <div>
                        <Link
                            href={item.href ?? "#"}
                            className={
                                // base link styles
                                "block text-sm text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400"
                            }
                            aria-current={undefined}
                            // indent by level (level 1 => 0px, level 2 => 12px, level 3 => 24px, ...)
                            style={{ paddingLeft: `${Math.max(0, item.level - 1) * 12}px` }}
                        >
                            {item.title}
                        </Link>
                    </div>

                    {item.children && item.children.length > 0 && (
                        <div className="mt-1">{renderMenuItems(item.children)}</div>
                    )}
                </li>
            ))}
        </ul>
    )
}

export default function IntroLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
            <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-8">
                    {/* Sidebar */}
                    <aside className="order-2 lg:order-1">
                        <div className="sticky top-6">
                            <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                                <h3 className="text-sm font-semibold mb-3 text-gray-900 dark:text-white">Tài liệu</h3>
                                <nav aria-label="Documentation menu">{renderMenuItems(docMenu)}</nav>
                            </div>

                            <div className="mt-4 p-3 text-xs text-gray-500 dark:text-gray-400">
                                <p>Tip: Sử dụng thanh điều hướng để chuyển giữa các phần của tài liệu.</p>
                            </div>
                        </div>
                    </aside>

                    {/* Main content area */}
                    <main className="order-1 lg:order-2">
                        <section className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
                            {children}
                        </section>
                    </main>
                </div>
            </div>
        </div>
    )
}