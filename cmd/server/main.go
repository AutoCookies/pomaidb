// File: cmd/cache-server/main.go
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/api"
	"github.com/AutoCookies/pomai-cache/internal/cache"
)

func main() {
	// Config via env or flags
	var (
		portEnv           = getEnv("PORT", "8080")
		shardsEnv         = getEnv("CACHE_SHARDS", "32")
		perTenantCapacity = getEnv("PER_TENANT_CAPACITY_BYTES", "104857600") // 100MB default
		gracefulSec       = getEnv("GRACEFUL_SHUTDOWN_SEC", "10")
		cleanupSec        = getEnv("CLEANUP_INTERVAL_SEC", "60") // 1 minute

		addrFlag      = flag.String("addr", ":"+portEnv, "listen address")
		shardsFlag    = flag.Int("shards", atoiDefault(shardsEnv, 32), "shard count")
		perTenantFlag = flag.Int64("perTenantCapacity", atoi64Default(perTenantCapacity, 100*1024*1024), "per-user capacity bytes")
		gracefulFlag  = flag.Int("graceful", atoiDefault(gracefulSec, 10), "graceful shutdown seconds")
		cleanupFlag   = flag.Int("cleanup", atoiDefault(cleanupSec, 60), "cleanup interval seconds")
	)
	flag.Parse()

	tenants := cache.NewTenantManager(*shardsFlag, *perTenantFlag)

	// ✅ Start background cleanup for all user stores
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		ticker := time.NewTicker(time.Duration(*cleanupFlag) * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				log.Println("[CLEANUP] Stopped")
				return
			case <-ticker.C:
				// Cleanup expired keys for all users
				userIDs := tenants.ListTenants()
				totalCleaned := 0

				for _, userID := range userIDs {
					store := tenants.GetStore(userID)
					cleaned := store.CleanupExpired() // ✅ Use public method
					totalCleaned += cleaned
				}

				if totalCleaned > 0 {
					log.Printf("[CLEANUP] Removed %d expired keys across %d users", totalCleaned, len(userIDs))
				}
			}
		}
	}()

	// Server requires X-User-Id header to identify user
	srv := api.NewServer(tenants, true)

	httpSrv := &http.Server{
		Addr:         *addrFlag,
		Handler:      srv.Router(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	capacityMB := *perTenantFlag / (1024 * 1024)
	log.Printf("Cache server starting on %s", *addrFlag)
	log.Printf("   - Per-user capacity: %dMB", capacityMB)
	log.Printf("   - Shard count: %d", *shardsFlag)
	log.Printf("   - Cleanup interval: %ds", *cleanupFlag)

	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server ListenAndServe: %v", err)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
	cancel() // Stop cleanup goroutine

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), time.Duration(*gracefulFlag)*time.Second)
	defer shutdownCancel()

	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	} else {
		log.Println("Server stopped gracefully")
	}
}

func getEnv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func atoiDefault(s string, d int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return d
}

func atoi64Default(s string, d int64) int64 {
	if v, err := strconv.ParseInt(s, 10, 64); err == nil {
		return v
	}
	return d
}
