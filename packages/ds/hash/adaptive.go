package hash

const (
	// Ngưỡng chuyển đổi: Dưới 128 phần tử dùng Flat (tiết kiệm RAM, CPU Cache tốt).
	// Trên 128 dùng Map (truy cập O(1)).
	MaxFlatSize = 128
)

// KV lưu trữ một cặp key-value trong Flat mode
type KV struct {
	Field string
	Value []byte
}

// HashObject là interface ẩn đi sự phức tạp của việc chuyển đổi
type HashObject interface {
	Get(field string) ([]byte, bool)
	Set(field string, value []byte) // Có thể trigger upgrade
	Delete(field string) bool
	Len() int
	GetAll() map[string][]byte
	IsFlat() bool // Debug/Stats
}

// ---------------------------------------------------------
// 1. FLAT HASH (Small Data - High Performance RAM)
// Lưu trữ như một mảng liền mạch. Không overhead của Map buckets.
// ---------------------------------------------------------
type FlatHash struct {
	entries []KV
}

func NewFlatHash() *FlatHash {
	return &FlatHash{
		entries: make([]KV, 0, 4), // Pre-alloc nhỏ
	}
}

func (f *FlatHash) IsFlat() bool { return true }

func (f *FlatHash) Len() int { return len(f.entries) }

func (f *FlatHash) Get(field string) ([]byte, bool) {
	// Linear Search: Với N < 128, việc này nhanh hơn Hashing + Map Lookups
	// nhờ Cache Locality (dữ liệu nằm liền nhau trong RAM).
	for i := range f.entries {
		if f.entries[i].Field == field {
			return f.entries[i].Value, true
		}
	}
	return nil, false
}

func (f *FlatHash) GetAll() map[string][]byte {
	res := make(map[string][]byte, len(f.entries))
	for _, kv := range f.entries {
		res[kv.Field] = kv.Value
	}
	return res
}

func (f *FlatHash) Set(field string, value []byte) {
	// Update existing
	for i := range f.entries {
		if f.entries[i].Field == field {
			f.entries[i].Value = value
			return
		}
	}
	// Append new
	f.entries = append(f.entries, KV{Field: field, Value: value})
}

func (f *FlatHash) Delete(field string) bool {
	for i := range f.entries {
		if f.entries[i].Field == field {
			// Xóa bằng cách swap phần tử cuối lên và cắt đuôi (Unordered)
			// Hoặc shift nếu cần thứ tự (ở đây Hash không cần thứ tự)
			lastIdx := len(f.entries) - 1
			f.entries[i] = f.entries[lastIdx] // Copy thằng cuối đè lên
			f.entries[lastIdx] = KV{}         // Clear để tránh memory leak
			f.entries = f.entries[:lastIdx]
			return true
		}
	}
	return false
}

// ---------------------------------------------------------
// 2. MAP HASH (Large Data - O(1) Access)
// Dùng native map khi dữ liệu lớn
// ---------------------------------------------------------
type MapHash struct {
	m map[string][]byte
}

func NewMapHash() *MapHash {
	return &MapHash{
		m: make(map[string][]byte),
	}
}

func (m *MapHash) IsFlat() bool { return false }
func (m *MapHash) Len() int     { return len(m.m) }

func (m *MapHash) Get(field string) ([]byte, bool) {
	val, ok := m.m[field]
	return val, ok
}

func (m *MapHash) Set(field string, value []byte) {
	m.m[field] = value
}

func (m *MapHash) Delete(field string) bool {
	_, ok := m.m[field]
	if ok {
		delete(m.m, field)
	}
	return ok
}

func (m *MapHash) GetAll() map[string][]byte {
	// Copy để an toàn concurrency bên ngoài
	res := make(map[string][]byte, len(m.m))
	for k, v := range m.m {
		res[k] = v
	}
	return res
}

// Upgrade chuyển đổi từ Flat -> Map
func (f *FlatHash) ToMap() *MapHash {
	mh := NewMapHash()
	// Pre-alloc map
	mh.m = make(map[string][]byte, len(f.entries)*2)
	for _, kv := range f.entries {
		mh.m[kv.Field] = kv.Value
	}
	return mh
}
