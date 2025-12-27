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

	"github.com/joho/godotenv"

	// [Adapter Imports]
	httpadapter "github.com/AutoCookies/pomai-cache/internal/adapter/httpadapter"
	"github.com/AutoCookies/pomai-cache/internal/adapter/persistence"

	// Import Firebase
	firebase_setup "github.com/AutoCookies/pomai-cache/internal/adapter/firebase"
	"github.com/AutoCookies/pomai-cache/internal/adapter/postgresql"

	// Import Auth Adapters (Đảm bảo bạn đã tạo các package này từ các bước trước)
	"github.com/AutoCookies/pomai-cache/internal/adapter/email"
	"github.com/AutoCookies/pomai-cache/internal/adapter/token"

	// [Core Imports]
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
	"github.com/AutoCookies/pomai-cache/internal/core/services"
	"github.com/AutoCookies/pomai-cache/internal/engine"
)

func main() {
	// ============================================================
	// 0. Load Environment Variables
	// ============================================================
	// Cố gắng load file .env. Nếu không thấy (ví dụ chạy trên Docker/Prod đã có env thật) thì bỏ qua lỗi.
	// Hàm này sẽ tìm file .env ở thư mục hiện tại chạy lệnh terminal.
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found or failed to load, relying on system env vars")
	} else {
		log.Println("Loaded environment variables from .env")
	}
	// ============================================================
	// 1. Configuration
	// ============================================================
	var (
		portEnv     = getEnv("PORT", "8080")
		shardsEnv   = getEnv("CACHE_SHARDS", "32")
		gracefulSec = getEnv("GRACEFUL_SHUTDOWN_SEC", "10")

		// Cache Config
		perTenantCapacity = getEnv("PER_TENANT_CAPACITY_BYTES", "104857600")
		cleanupSec        = getEnv("CLEANUP_INTERVAL_SEC", "60")

		// TTL & Bloom
		enableAdaptiveTTL = getEnv("ENABLE_ADAPTIVE_TTL", "true")
		adaptiveMinTTL    = getEnv("ADAPTIVE_MIN_TTL", "1m")
		adaptiveMaxTTL    = getEnv("ADAPTIVE_MAX_TTL", "1h")
		enableBloom       = getEnv("ENABLE_BLOOM_FILTER", "true")
		bloomSize         = getEnv("BLOOM_SIZE", "10000000")
		bloomK            = getEnv("BLOOM_K", "4")

		// Persistence
		persistenceType   = getEnv("PERSISTENCE_TYPE", "file")
		dataDir           = getEnv("DATA_DIR", "./data")
		snapshotInterval  = getEnv("SNAPSHOT_INTERVAL", "5m")
		enableWriteBehind = getEnv("ENABLE_WRITE_BEHIND", "false")
		writeBufferSize   = getEnv("WRITE_BUFFER_SIZE", "1000")
		flushInterval     = getEnv("FLUSH_INTERVAL", "5s")

		// Auth Config
		tokenSecret  = getEnv("JWT_ACCESS_SECRET", "12345678901234567890123456789012") // 32 chars min
		resendAPIKey = os.Getenv("RESEND_API_KEY")
		resendFrom   = getEnv("RESEND_FROM_EMAIL", "Cookiescooker <no-reply@cookiescooker.click>")

		// Flags
		addrFlag      = flag.String("addr", ":"+portEnv, "listen address")
		shardsFlag    = flag.Int("shards", atoiDefault(shardsEnv, 32), "shard count")
		perTenantFlag = flag.Int64("perTenantCapacity", atoi64Default(perTenantCapacity, 100*1024*1024), "per-user capacity bytes")
		gracefulFlag  = flag.Int("graceful", atoiDefault(gracefulSec, 10), "graceful shutdown seconds")
		cleanupFlag   = flag.Int("cleanup", atoiDefault(cleanupSec, 60), "cleanup interval seconds")
	)

	flag.Parse()

	log.Printf("🚀 Pomegranate Server starting on %s", *addrFlag)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	postgresql.InitializePostgreSQL() // Đảm bảo sử dụng Singleton để khởi tạo một lần
	postgresPool := postgresql.GetPostgresPool()

	// ============================================================
	// 2. Initialize Infrastructure (Firebase)
	// ============================================================
	log.Println("[INIT] Connecting to Firebase...")
	firebase_setup.Initialize()
	firestoreClient := firebase_setup.GetDB()

	// ============================================================
	// 3. Initialize Cache Engine
	// ============================================================
	log.Println("[INIT] Starting Cache Engine...")
	tenants := engine.NewTenantManager(*shardsFlag, *perTenantFlag)
	defaultStore := tenants.GetStore("default")

	if enableAdaptiveTTL == "true" {
		min, _ := time.ParseDuration(adaptiveMinTTL)
		max, _ := time.ParseDuration(adaptiveMaxTTL)
		defaultStore.EnableAdaptiveTTL(min, max)
	}
	if enableBloom == "true" {
		size, _ := strconv.ParseUint(bloomSize, 10, 64)
		k, _ := strconv.ParseUint(bloomK, 10, 64)
		defaultStore.EnableBloomFilter(size, k)
	}

	// ============================================================
	// 4. Initialize Persistence (Cache Layer)
	// ============================================================
	var persister ports.Persister
	var err error

	switch persistenceType {
	case "file":
		persister, err = persistence.NewFilePersister(filepath.Join(dataDir, "cache"))
		if err != nil {
			log.Fatalf("Failed to create file persister: %v", err)
		}
		if fp, ok := persister.(*persistence.FilePersister); ok {
			_ = fp.Restore(defaultStore)
			interval, _ := time.ParseDuration(snapshotInterval)
			fp.StartPeriodicSnapshot(defaultStore, interval)
		}
	case "wal":
		persister, err = persistence.NewWALPersister(filepath.Join(dataDir, "wal.log"))
		if err != nil {
			log.Fatalf("Failed to create WAL persister: %v", err)
		}
		if wp, ok := persister.(*persistence.WALPersister); ok {
			_ = wp.Restore(defaultStore)
		}
	default:
		persister = persistence.NewNoOpPersister()
	}

	if enableWriteBehind == "true" && persistenceType != "noop" {
		bs, _ := strconv.Atoi(writeBufferSize)
		fi, _ := time.ParseDuration(flushInterval)
		wb := persistence.NewWriteBehindBuffer(bs, fi, persister)
		wb.Start(ctx)
		defer wb.Close()
	} else {
		defer persister.Close()
	}

	// ============================================================
	// 5. Initialize Auth Stack (Wiring)
	// ============================================================
	log.Println("[INIT] Setting up Auth Stack...")

	// 5.1 Repositories
	authRepo := persistence.NewFirestoreAuthRepo(firestoreClient)
	verifyRepo := persistence.NewFirestoreVerificationRepo(firestoreClient)

	// 5.2 Token Maker
	tokenMaker, err := token.NewJWTMaker(tokenSecret)
	if err != nil {
		log.Fatalf("Failed to create token maker: %v", err)
	}

	// 5.3 Email Sender
	var emailSender ports.EmailSender
	if resendAPIKey == "" {
		log.Println("⚠️ RESEND_API_KEY missing. Email sending will fail/mock.")
		// Fallback to mock/dummy adapter if needed
		emailSender, _ = email.NewResendAdapter("mock_key", resendFrom)
	} else {
		emailSender, err = email.NewResendAdapter(resendAPIKey, resendFrom)
		if err != nil {
			log.Fatalf("Failed to create email adapter: %v", err)
		}
	}

	// 5.4 Service
	authService := services.NewAuthService(authRepo, verifyRepo, tokenMaker, emailSender)

	// 5.5 Handler
	authHandler := httpadapter.NewAuthHandler(authService)

	// ============================================================
	// 6. Initialize HTTP Server
	// ============================================================
	requireAuth := os.Getenv("REQUIRE_AUTH") == "true"

	// Khởi tạo APIKeyRepo và APIKeyService
	apiKeyRepo := persistence.NewAPIKeyRepo(postgresPool)  // Tạo repo APIKey
	apiKeyService := services.NewAPIKeyService(apiKeyRepo) // Dùng APIKeyRepo trong service

	// Truyền thêm apiKeyService vào NewServer
	srv := httpadapter.NewServer(tenants, authHandler, tokenMaker, requireAuth, apiKeyService)

	httpSrv := &http.Server{
		Addr:         *addrFlag,
		Handler:      srv.Router(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Listen error: %v", err)
		}
	}()

	go func() {
		ticker := time.NewTicker(time.Duration(*cleanupFlag) * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Duyệt qua tất cả tenant và xóa các key hết hạn
				for _, uid := range tenants.ListTenants() {
					tenants.GetStore(uid).CleanupExpired()
				}
			}
		}
	}()

	// ============================================================
	// 7. Graceful Shutdown
	// ============================================================
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), time.Duration(*gracefulFlag)*time.Second)
	defer shutdownCancel()

	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("Shutdown error: %v", err)
	}

	log.Println("Bye!")
}

// Helper Functions
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
