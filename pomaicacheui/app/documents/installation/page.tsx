"use client";

import React, { useState } from "react";

const CodeBlock: React.FC<{ language?: string; code: string }> = ({ language = "bash", code }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            setCopied(false);
        }
    };

    return (
        <div className="relative bg-gray-900 text-gray-100 rounded-lg overflow-hidden shadow-sm">
            <pre className="p-4 text-xs leading-5 overflow-x-auto">
                <code className={`language-${language}`}>{code}</code>
            </pre>
            <button
                onClick={handleCopy}
                className="absolute top-2 right-2 bg-white/10 hover:bg-white/20 text-xs px-2 py-1 rounded"
                aria-label="Copy code"
            >
                {copied ? "Copied" : "Copy"}
            </button>
        </div>
    );
};

export default function InstallPage() {
    const npmInstall = `# npm
npm install @pomai/sdk

# yarn
yarn add @pomai/sdk

# pnpm
pnpm add @pomai/sdk`;
    const jsExample = `// ESM
import PomaiCache from "@pomai/sdk";

const client = new PomaiCache({ apiKey: "KID.YOUR_SECRET" });
await client.set("hello", "world", { ttl: 60 });
const v = await client.get("hello"); // "world"`;

    const goGet = `# go (module aware)
go get github.com/AutoCookies/sdk@v0.1.0

# or use latest
go get github.com/AutoCookies/sdk`;
    const goExample = `package main

import (
  "context"
  "fmt"
  "time"

  "github.com/AutoCookies/sdk"
)

func main() {
  client := sdk.New("KID.YOUR_SECRET") // only API key required (SDK talks to public gateway)
  ctx := context.Background()

  _ = client.Set(ctx, "hello", []byte("world"), 60*time.Second)
  v, found, _ := client.Get(ctx, "hello")
  if found {
    fmt.Println(string(v))
  }
}`;

    return (
        <div className="max-w-4xl mx-auto py-12 px-6">
            <header className="mb-8">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Tải và cài đặt Pomai Cache SDK</h1>
                <p className="mt-2 text-gray-600 dark:text-gray-300">
                    Hiện tại chúng tôi cung cấp SDK cho JavaScript (npm) và Go. Bạn chỉ cần cài package và truyền API key
                    (được tạo trong dashboard) — không cần biết địa chỉ nội bộ của hạ tầng.
                </p>
            </header>

            <section className="mb-8">
                <h2 className="text-lg font-semibold mb-3">JavaScript / TypeScript (npm)</h2>
                <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                    Package tên: <code>@pomai/sdk</code> (thay đổi tuỳ repo khi publish). Chọn 1 trong các lệnh dưới để cài.
                </p>
                <CodeBlock language="bash" code={npmInstall} />
                <div className="mt-4">
                    <h3 className="font-medium mb-2">Ví dụ nhanh (ESM)</h3>
                    <CodeBlock language="js" code={jsExample} />
                </div>
                <p className="mt-3 text-xs text-gray-500">
                    Ghi chú: nếu sử dụng CommonJS, import tương ứng (require) hoặc build tool sẽ xử lý. Luôn giữ API key an toàn
                    (env var / secret manager).
                </p>
            </section>

            <section className="mb-8">
                <h2 className="text-lg font-semibold mb-3">Go (go get)</h2>
                <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                    Module path: <code>github.com/AutoCookies/sdk</code> (ví dụ). Cài bằng <code>go get</code> với tag phiên bản.
                </p>
                <CodeBlock language="bash" code={goGet} />
                <div className="mt-4">
                    <h3 className="font-medium mb-2">Ví dụ nhanh (Go)</h3>
                    <CodeBlock language="go" code={goExample} />
                </div>
                <p className="mt-3 text-xs text-gray-500">
                    Ghi chú: SDK Go khởi tạo chỉ với API key (ví dụ <code>sdk.New("KID.YOUR_SECRET")</code>). SDK sẽ giao tiếp với
                    gateway công khai của chúng tôi — bạn không cần biết URL nội bộ.
                </p>
            </section>

            <section className="mb-8">
                <h2 className="text-lg font-semibold mb-3">Lời khuyên bảo mật</h2>
                <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-300 space-y-2">
                    <li>Không commit API key lên repository công khai — dùng biến môi trường hoặc secret manager.</li>
                    <li>API key chỉ hiển thị một lần khi tạo; lưu lại cẩn thận. Nếu nghi ngờ, rotate hoặc revoke key.</li>
                    <li>Kiểm tra phiên bản SDK trước khi dùng (để tránh breaking changes): cài theo tag semver.</li>
                </ul>
            </section>

            <footer className="text-sm text-gray-500">
                <p>Nếu bạn muốn SDK cho ngôn ngữ khác (Python, Java), chúng tôi sẽ bổ sung sớm. Cần hỗ trợ publish hay ví dụ thêm thì báo nhé.</p>
            </footer>
        </div>
    );
}