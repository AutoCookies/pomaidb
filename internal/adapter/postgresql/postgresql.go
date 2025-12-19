package postgresql

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
)

var (
	pool *pgxpool.Pool
	once sync.Once
)

// InitializePostgreSQL khởi tạo kết nối PostgreSQL (Singleton)
func InitializePostgreSQL() {
	once.Do(func() {
		// Tạo context
		ctx := context.Background()

		// Lấy chuỗi kết nối từ biến môi trường
		dbURL := os.Getenv("POSTGRESQL_HOST")
		if dbURL == "" {
			log.Fatal("POSTGRESQL_HOST not set in environment or .env file")
		}

		// Khởi tạo Connection Pool
		var err error
		pool, err = pgxpool.New(ctx, dbURL)
		if err != nil {
			log.Fatalf("Unable to connect to database: %v", err)
		}

		// Kiểm tra kết nối
		err = pool.Ping(ctx)
		if err != nil {
			log.Fatalf("Ping error: %v", err)
		}

		fmt.Println("[PostgreSQL] Initialized successfully")
	})
}

// GetPostgresPool trả về instance của Pool cho PostgreSQL
func GetPostgresPool() *pgxpool.Pool {
	if pool == nil {
		InitializePostgreSQL()
	}
	return pool
}
