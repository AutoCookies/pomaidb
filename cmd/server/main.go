// File: cmd/cache-server/main.go
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/api"
	"github.com/AutoCookies/pomai-cache/internal/cache"
	"github.com/AutoCookies/pomai-cache/internal/persistence"
)

func main() {
	// ============================================================
	// Configuration
	// ============================================================
	var (
		// Server config
		portEnv     = getEnv("PORT", "8080")
		shardsEnv   = getEnv("CACHE_SHARDS", "32")
		gracefulSec = getEnv("GRACEFUL_SHUTDOWN_SEC", "10")

		// Cache config
		perTenantCapacity = getEnv("PER_TENANT_CAPACITY_BYTES", "104857600") // 100MB
		cleanupSec        = getEnv("CLEANUP_INTERVAL_SEC", "60")             // 1 minute

		// Adaptive TTL config
		enableAdaptiveTTL = getEnv("ENABLE_ADAPTIVE_TTL", "true")
		adaptiveMinTTL    = getEnv("ADAPTIVE_MIN_TTL", "1m")
		adaptiveMaxTTL    = getEnv("ADAPTIVE_MAX_TTL", "1h")

		// Bloom filter config
		enableBloom = getEnv("ENABLE_BLOOM_FILTER", "true")
		bloomSize   = getEnv("BLOOM_SIZE", "10000000") // 10M bits (~1. 2MB)
		bloomK      = getEnv("BLOOM_K", "4")           // 4 hash functions

		// Persistence config
		persistenceType  = getEnv("PERSISTENCE_TYPE", "file") // "file", "wal", "noop"
		dataDir          = getEnv("DATA_DIR", "./data")
		snapshotInterval = getEnv("SNAPSHOT_INTERVAL", "5m")

		// Write-behind config
		enableWriteBehind = getEnv("ENABLE_WRITE_BEHIND", "false")
		writeBufferSize   = getEnv("WRITE_BUFFER_SIZE", "1000")
		flushInterval     = getEnv("FLUSH_INTERVAL", "5s")

		// Flags
		addrFlag      = flag.String("addr", ":"+portEnv, "listen address")
		shardsFlag    = flag.Int("shards", atoiDefault(shardsEnv, 32), "shard count")
		perTenantFlag = flag.Int64("perTenantCapacity", atoi64Default(perTenantCapacity, 100*1024*1024), "per-user capacity bytes")
		gracefulFlag  = flag.Int("graceful", atoiDefault(gracefulSec, 10), "graceful shutdown seconds")
		cleanupFlag   = flag.Int("cleanup", atoiDefault(cleanupSec, 60), "cleanup interval seconds")
	)
	flag.Parse()

	log.Println("=== Pomegranate Cache Server ===")
	log.Printf("Version: 1.0.0")
	log.Printf("Server: %s", *addrFlag)
	log.Printf("Per-user capacity: %dMB", *perTenantFlag/(1024*1024))
	log.Printf("Shard count: %d", *shardsFlag)

	// ============================================================
	// Initialize Tenant Manager & Default Store
	// ============================================================
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tenants := cache.NewTenantManager(*shardsFlag, *perTenantFlag)
	defaultStore := tenants.GetStore("default")

	// ============================================================
	// Feature 1: Adaptive TTL
	// ============================================================
	if enableAdaptiveTTL == "true" {
		minDuration, err := time.ParseDuration(adaptiveMinTTL)
		if err != nil {
			log.Fatalf("Invalid ADAPTIVE_MIN_TTL: %v", err)
		}
		maxDuration, err := time.ParseDuration(adaptiveMaxTTL)
		if err != nil {
			log.Fatalf("Invalid ADAPTIVE_MAX_TTL: %v", err)
		}

		defaultStore.EnableAdaptiveTTL(minDuration, maxDuration)
		log.Printf("[ADAPTIVE TTL] Enabled:  min=%v, max=%v", minDuration, maxDuration)
	} else {
		log.Println("[ADAPTIVE TTL] Disabled")
	}

	// ============================================================
	// Feature 2: Bloom Filter
	// ============================================================
	if enableBloom == "true" {
		size, err := strconv.ParseUint(bloomSize, 10, 64)
		if err != nil {
			log.Fatalf("Invalid BLOOM_SIZE:  %v", err)
		}
		k, err := strconv.ParseUint(bloomK, 10, 64)
		if err != nil {
			log.Fatalf("Invalid BLOOM_K: %v", err)
		}

		defaultStore.EnableBloomFilter(size, k)
		log.Printf("[BLOOM FILTER] Enabled: size=%d bits (~%. 2fMB), k=%d hash functions",
			size, float64(size)/8/1024/1024, k)
	} else {
		log.Println("[BLOOM FILTER] Disabled")
	}

	// ============================================================
	// Feature 3: Persistence
	// ============================================================
	var persister persistence.Persister
	var err error

	switch persistenceType {
	case "file":
		persister, err = persistence.NewFilePersister(filepath.Join(dataDir, "cache"))
		if err != nil {
			log.Fatalf("[PERSISTENCE] Failed to create file persister: %v", err)
		}

		// Restore from snapshot
		if err := persister.Restore(defaultStore); err != nil {
			log.Printf("[PERSISTENCE] Warning:  Restore failed: %v", err)
		}

		// Start periodic snapshots
		if fp, ok := persister.(*persistence.FilePersister); ok {
			interval, err := time.ParseDuration(snapshotInterval)
			if err != nil {
				log.Fatalf("[PERSISTENCE] Invalid SNAPSHOT_INTERVAL: %v", err)
			}
			fp.StartPeriodicSnapshot(defaultStore, interval)
			log.Printf("[PERSISTENCE] File persister enabled:  interval=%v, path=%s",
				interval, filepath.Join(dataDir, "cache", "snapshot. gob"))
		}

	case "wal":
		persister, err = persistence.NewWALPersister(filepath.Join(dataDir, "wal. log"))
		if err != nil {
			log.Fatalf("[PERSISTENCE] Failed to create WAL persister: %v", err)
		}

		// Restore from WAL + snapshot
		if err := persister.Restore(defaultStore); err != nil {
			log.Printf("[PERSISTENCE] Warning: WAL restore failed: %v", err)
		}

		log.Printf("[PERSISTENCE] WAL persister enabled:  path=%s", filepath.Join(dataDir, "wal.log"))

	default:
		persister = persistence.NewNoOpPersister()
		log.Println("[PERSISTENCE] Disabled (using noop)")
	}

	defer func() {
		if persistenceType != "noop" {
			log.Println("[PERSISTENCE] Creating final snapshot...")
			if err := persister.Snapshot(defaultStore); err != nil {
				log.Printf("[PERSISTENCE] Warning: Final snapshot failed: %v", err)
			} else {
				log.Println("[PERSISTENCE] Final snapshot completed")
			}
		}
		persister.Close()
	}()

	// ============================================================
	// Feature 4: Write-Behind Buffer
	// ============================================================
	var writeBehind *persistence.WriteBehindBuffer

	if enableWriteBehind == "true" && persistenceType != "noop" {
		bufferSize, err := strconv.Atoi(writeBufferSize)
		if err != nil {
			log.Fatalf("[WRITE-BEHIND] Invalid WRITE_BUFFER_SIZE: %v", err)
		}
		interval, err := time.ParseDuration(flushInterval)
		if err != nil {
			log.Fatalf("[WRITE-BEHIND] Invalid FLUSH_INTERVAL: %v", err)
		}

		writeBehind = persistence.NewWriteBehindBuffer(bufferSize, interval, persister)
		writeBehind.Start(ctx)

		defer writeBehind.Close()

		log.Printf("[WRITE-BEHIND] Enabled: bufferSize=%d, interval=%v", bufferSize, interval)
	} else {
		log.Println("[WRITE-BEHIND] Disabled")
	}

	// ============================================================
	// Feature 5: Background Cleanup
	// ============================================================
	go func() {
		ticker := time.NewTicker(time.Duration(*cleanupFlag) * time.Second)
		defer ticker.Stop()

		log.Printf("[CLEANUP] Started:  interval=%ds", *cleanupFlag)

		for {
			select {
			case <-ctx.Done():
				log.Println("[CLEANUP] Stopped")
				return
			case <-ticker.C:
				userIDs := tenants.ListTenants()
				totalCleaned := 0

				for _, userID := range userIDs {
					store := tenants.GetStore(userID)
					cleaned := store.CleanupExpired()
					totalCleaned += cleaned
				}

				if totalCleaned > 0 {
					log.Printf("[CLEANUP] Removed %d expired keys across %d users", totalCleaned, len(userIDs))
				}

				// Rebuild bloom filter periodically if many keys were removed
				if enableBloom == "true" && totalCleaned > 1000 {
					defaultStore.RebuildBloomFilter()
				}
			}
		}
	}()

	// ============================================================
	// HTTP Server
	// ============================================================
	srv := api.NewServer(tenants, true) // true = require X-User-Id header

	httpSrv := &http.Server{
		Addr:         *addrFlag,
		Handler:      srv.Router(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Println("=================================")
		log.Printf("🚀 Server listening on %s", *addrFlag)
		log.Println("=================================")

		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[SERVER] ListenAndServe error: %v", err)
		}
	}()

	// ============================================================
	// Graceful Shutdown
	// ============================================================
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit

	log.Printf("Received signal:  %v", sig)
	log.Println("=================================")
	log.Println("Shutting down gracefully...")
	log.Println("=================================")

	// Stop background tasks
	cancel()

	// Shutdown HTTP server
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), time.Duration(*gracefulFlag)*time.Second)
	defer shutdownCancel()

	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("[SERVER] Shutdown error:  %v", err)
	} else {
		log.Println("[SERVER] Stopped gracefully")
	}

	// Print final stats
	stats := defaultStore.Stats()
	log.Println("=================================")
	log.Println("Final Statistics:")
	log.Printf("  Total Hits:       %d", stats.Hits)
	log.Printf("  Total Misses:    %d", stats.Misses)
	log.Printf("  Hit Rate:        %.2f%%", float64(stats.Hits)/float64(stats.Hits+stats.Misses)*100)
	log.Printf("  Items:            %d", stats.Items)
	log.Printf("  Bytes:           %d (%.2fMB)", stats.Bytes, float64(stats.Bytes)/1024/1024)
	log.Printf("  Evictions:       %d", stats.Evictions)

	if stats.BloomEnabled {
		bloomStats := defaultStore.GetBloomStats()
		log.Printf("  Bloom Avoided:   %d lookups", bloomStats.Avoided)
		log.Printf("  Bloom FP Rate:   %.2f%%", bloomStats.FalsePositiveRate)
	}

	if writeBehind != nil {
		wbStats := writeBehind.Stats()
		log.Printf("  Write-Behind:")
		log.Printf("    Total Writes:   %d", wbStats.TotalWrites)
		log.Printf("    Total Flushes: %d", wbStats.TotalFlushes)
		log.Printf("    Failed:         %d", wbStats.FailedFlushes)
	}

	log.Println("=================================")
	log.Println("👋 Goodbye!")
}

// ============================================================
// Helper Functions
// ============================================================

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func atoiDefault(s string, defaultValue int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return defaultValue
}

func atoi64Default(s string, defaultValue int64) int64 {
	if v, err := strconv.ParseInt(s, 10, 64); err == nil {
		return v
	}
	return defaultValue
}
