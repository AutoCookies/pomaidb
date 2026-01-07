package ports

import "time"

// 1. WriteOp: Struct dữ liệu nằm ở Core để cả Engine và Adapter đều hiểu được
type WriteOp struct {
	Key   string
	Value []byte
	TTL   time.Duration
}

// 2. Persister: Interface quy định các hành động lưu trữ
type Persister interface {
	// Persist ghi 1 key
	Persist(key string, value []byte) error

	// PersistBatch ghi nhiều key (quan trọng cho WriteBehind)
	PersistBatch(ops []WriteOp) error

	// Load đọc dữ liệu
	Load(key string) ([]byte, error)

	// Close đóng kết nối
	Close() error
}
