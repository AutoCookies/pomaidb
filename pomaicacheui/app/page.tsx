"use client";
import React, { useState, useEffect } from "react";
import {
  Zap, Server, Shield, Activity, ArrowRight,
  Terminal, Check, Cpu, Globe, Lock
} from "lucide-react";
import Link from "next/link";

// --- ASSETS ---
// Bạn hãy import đường dẫn ảnh logo thực tế của bạn vào đây
// Ví dụ: import logoImg from "../assets/pomai-cache-logo.jpg";
const LOGO_URL = "/logo.png";

const PomaiCacheLanding = () => {
  const [scrolled, setScrolled] = useState(false);

  // Hiệu ứng thanh nav khi scroll
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-[#F9F7F2] font-sans text-slate-800 selection:bg-[#4B8B8B] selection:text-white">

      <nav className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${scrolled ? "bg-white/90 backdrop-blur-md shadow-sm py-3" : "bg-transparent py-5"}`}>
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src={LOGO_URL} alt="Pomai Cache Logo" className="w-10 h-10 rounded-lg shadow-md object-cover" />
            <span className="text-xl font-bold tracking-tight text-slate-900">Pomai <span className="text-[#4B8B8B]">Cache</span></span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <a href="#features" className="hover:text-[#4B8B8B] transition-colors">Features</a>
            <a href="#performance" className="hover:text-[#4B8B8B] transition-colors">Benchmarks</a>
            <a href="/documents" className="hover:text-[#4B8B8B] transition-colors">Documentation</a>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/signin" className="hidden md:block text-sm font-bold text-slate-700 hover:text-[#4B8B8B]">Sign In</Link>
            <button className="bg-[#4B8B8B] text-white px-5 py-2.5 rounded-lg text-sm font-bold shadow-lg shadow-[#4B8B8B]/20 hover:bg-[#3a6f6f] hover:-translate-y-0.5 transition-all">
              Get API Key
            </button>
          </div>
        </div>
      </nav>

      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 grid lg:grid-cols-2 gap-16 items-center">
          <div className="relative z-10 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#4B8B8B]/10 text-[#4B8B8B] text-xs font-bold uppercase tracking-widest mb-6 border border-[#4B8B8B]/20">
              <Zap size={14} /> v1.0 Stable Release
            </div>
            <h1 className="text-5xl lg:text-7xl font-extrabold text-slate-900 leading-[1.1] mb-6">
              Extreme <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#4B8B8B] to-emerald-400">Performance</span> <br />
              Data Caching.
            </h1>
            <p className="text-lg text-slate-600 mb-8 max-w-xl leading-relaxed">
              Stop waiting for database queries. Pomai Cache delivers sub-millisecond latency for your high-load applications. Distributed, atomic, and built for scale.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <button className="flex items-center justify-center gap-2 bg-[#E14D58] text-white px-8 py-4 rounded-xl font-bold text-lg shadow-xl shadow-[#E14D58]/30 hover:bg-[#c93b45] hover:-translate-y-1 transition-all">
                Start for Free <ArrowRight size={20} />
              </button>
              <button className="flex items-center justify-center gap-2 bg-white text-slate-700 border border-slate-200 px-8 py-4 rounded-xl font-bold text-lg hover:border-[#4B8B8B] hover:text-[#4B8B8B] transition-all">
                View Benchmarks
              </button>
            </div>

            <div className="mt-10 flex items-center gap-6 text-sm text-slate-500 font-medium">
              <span className="flex items-center gap-2"><Check size={16} className="text-[#4B8B8B]" /> 99.99% Uptime</span>
              <span className="flex items-center gap-2"><Check size={16} className="text-[#4B8B8B]" /> Global CDN</span>
            </div>
          </div>

          {/* Right Visual (Logo & Terminal) */}
          <div className="relative z-10">
            {/* Abstract Background Blob */}
            <div className="absolute -top-20 -right-20 w-96 h-96 bg-[#4B8B8B]/20 rounded-full blur-3xl opacity-50 animate-pulse"></div>

            <div className="relative bg-[#1e293b] rounded-2xl shadow-2xl border border-slate-700/50 overflow-hidden transform rotate-1 hover:rotate-0 transition-transform duration-500">
              {/* Window Header */}
              <div className="bg-[#0f172a] px-4 py-3 flex items-center gap-2 border-b border-slate-700">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <div className="ml-4 text-xs text-slate-400 font-mono">bash — pomai-cli</div>
              </div>

              {/* Terminal Content */}
              <div className="p-6 font-mono text-sm">
                <div className="flex gap-2 mb-4">
                  <span className="text-emerald-400">➜</span>
                  <span className="text-blue-400">~</span>
                  <span className="text-slate-300">pomai cache set --key="user:123" --ttl=60s</span>
                </div>
                <div className="text-slate-400 mb-6">
                  &gt; OK <span className="text-emerald-500">(0.04ms)</span>
                </div>

                <div className="flex gap-2 mb-4">
                  <span className="text-emerald-400">➜</span>
                  <span className="text-blue-400">~</span>
                  <span className="text-slate-300">pomai cache get --key="user:123"</span>
                </div>
                <div className="text-emerald-300 mb-2">
                  &#123;
                  <br />&nbsp;&nbsp;"id": 123,
                  <br />&nbsp;&nbsp;"name": "Justin Mason",
                  <br />&nbsp;&nbsp;"role": "admin",
                  <br />&nbsp;&nbsp;"plan": "enterprise"
                  <br />&#125;
                </div>
              </div>

              {/* Logo Overlay */}
              <div className="absolute -bottom-10 -right-10 w-48 h-48 opacity-20 rotate-12 pointer-events-none">
                <img src={LOGO_URL} alt="" className="w-full h-full object-contain" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- PERFORMANCE COMPARISON (THE "HOOK") --- */}
      <section id="performance" className="py-24 bg-white border-y border-gray-100">
        <div className="max-w-5xl mx-auto px-6 text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-extrabold text-slate-900 mb-4">Speed is a Feature.</h2>
          <p className="text-slate-500 text-lg">Don't let database bottlenecks kill your user experience.</p>
        </div>

        <div className="max-w-4xl mx-auto px-6">
          {/* Slow Bar */}
          <div className="mb-8">
            <div className="flex justify-between text-sm font-bold text-slate-500 mb-2">
              <span>Standard Database Query</span>
              <span className="text-red-500">~3.00s</span>
            </div>
            <div className="w-full h-12 bg-gray-100 rounded-lg overflow-hidden relative">
              <div className="absolute inset-y-0 left-0 bg-gray-300 w-full animate-pulse flex items-center justify-end px-4 text-gray-500 font-mono text-xs">
                Loading...
              </div>
            </div>
          </div>

          {/* Fast Bar */}
          <div className="relative">
            <div className="flex justify-between text-sm font-bold text-slate-800 mb-2">
              <span className="flex items-center gap-2 text-[#4B8B8B]">
                <img src={LOGO_URL} className="w-5 h-5 rounded" alt="" /> Pomai Cache
              </span>
              <span className="text-[#4B8B8B] text-lg">38ms</span>
            </div>
            <div className="w-full h-12 bg-gray-100 rounded-lg overflow-hidden relative shadow-inner">
              <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-[#4B8B8B] to-emerald-400 w-[4%] shadow-[0_0_20px_rgba(75,139,139,0.5)] flex items-center">
                {/* Flash effect */}
                <div className="w-full h-full bg-white/30 animate-ping"></div>
              </div>
            </div>
            <div className="mt-4 text-center">
              <span className="inline-block px-4 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold uppercase">
                🚀 80x Faster
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* --- FEATURES GRID --- */}
      <section id="features" className="py-24 bg-[#F9F7F2]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900">Why Developers Choose Pomai</h2>
            <div className="w-20 h-1 bg-[#4B8B8B] mx-auto mt-4 rounded-full"></div>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-xl transition-shadow border border-gray-100 group">
              <div className="w-14 h-14 bg-[#4B8B8B]/10 rounded-xl flex items-center justify-center text-[#4B8B8B] mb-6 group-hover:scale-110 transition-transform">
                <Cpu size={28} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Concurrency Native</h3>
              <p className="text-slate-600 leading-relaxed">
                Built to handle massive concurrent connections without blocking. Ideal for microservices and real-time apps.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-xl transition-shadow border border-gray-100 group">
              <div className="w-14 h-14 bg-[#E14D58]/10 rounded-xl flex items-center justify-center text-[#E14D58] mb-6 group-hover:scale-110 transition-transform">
                <Globe size={28} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Edge Distributed</h3>
              <p className="text-slate-600 leading-relaxed">
                Data is automatically replicated across nodes close to your users. Low latency, no matter where they are.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-xl transition-shadow border border-gray-100 group">
              <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center text-blue-600 mb-6 group-hover:scale-110 transition-transform">
                <Lock size={28} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Atomic Integrity</h3>
              <p className="text-slate-600 leading-relaxed">
                Thread-safe operations ensure your data remains consistent, even under extreme write loads.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="py-20 bg-[#1e293b] text-white overflow-hidden relative">
        <div className="absolute top-0 right-0 w-96 h-96 bg-[#4B8B8B] rounded-full blur-[128px] opacity-20"></div>

        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
            Ready to speed up your app?
          </h2>
          <p className="text-slate-400 text-lg mb-10">
            Join thousands of developers building faster, more scalable applications with Pomai Cache.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button className="w-full sm:w-auto bg-[#4B8B8B] text-white px-8 py-4 rounded-xl font-bold text-lg shadow-lg hover:bg-[#3a6f6f] transition-all">
              Get Started for Free
            </button>
            <button className="w-full sm:w-auto bg-transparent border border-slate-600 text-white px-8 py-4 rounded-xl font-bold text-lg hover:bg-white/10 transition-all">
              Read the Docs
            </button>
          </div>
        </div>
      </section>

      <footer className="bg-white py-12 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-[#4B8B8B] flex items-center justify-center text-white font-bold">P</div>
            <span className="font-bold text-slate-900">Pomai Ecosystem</span>
          </div>
          <div className="text-slate-500 text-sm">
            © 2024 Pomai Inc. Built for performance.
          </div>
          <div className="flex gap-6">
            <a href="#" className="text-slate-400 hover:text-[#4B8B8B]"><Globe size={20} /></a>
            <a href="#" className="text-slate-400 hover:text-[#4B8B8B]"><Terminal size={20} /></a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PomaiCacheLanding;