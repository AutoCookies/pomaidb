// File: internal/engine/core/entry.go
package common

import (
	"sync/atomic"
	"time"
)

type Entry struct {
	key        string
	value      []byte
	offset     uint32
	size       uint32
	expireAt   int64
	lastAccess atomic.Int64
	accesses   atomic.Uint64
	createdAt  int64
}

func NewEntry(key string, value []byte, ttl time.Duration) *Entry {
	now := time.Now().UnixNano()

	entry := &Entry{
		key:       key,
		value:     value,
		size:      uint32(len(value)),
		createdAt: now,
	}

	entry.lastAccess.Store(now)
	entry.accesses.Store(1)

	if ttl > 0 {
		entry.expireAt = time.Now().Add(ttl).UnixNano()
	}

	return entry
}

func (e *Entry) Key() string {
	return e.key
}

func (e *Entry) Value() []byte {
	return e.value
}

func (e *Entry) Size() int {
	return int(e.size)
}

func (e *Entry) IsExpired() bool {
	if e.expireAt == 0 {
		return false
	}
	return time.Now().UnixNano() > e.expireAt
}

func (e *Entry) IsExpiredAt(now int64) bool {
	if e.expireAt == 0 {
		return false
	}
	return now > e.expireAt
}

func (e *Entry) Touch() {
	now := time.Now().UnixNano()
	e.lastAccess.Store(now)
	e.accesses.Add(1)
}

func (e *Entry) TouchFast() uint64 {
	e.lastAccess.Store(time.Now().UnixNano())
	return e.accesses.Add(1)
}

func (e *Entry) TouchNoTime() uint64 {
	return e.accesses.Add(1)
}

func (e *Entry) Age() time.Duration {
	return time.Duration(time.Now().UnixNano() - e.createdAt)
}

func (e *Entry) AgeNano() int64 {
	return time.Now().UnixNano() - e.createdAt
}

func (e *Entry) Accesses() uint64 {
	return e.accesses.Load()
}

func (e *Entry) LastAccess() int64 {
	return e.lastAccess.Load()
}

func (e *Entry) CreatedAt() int64 {
	return e.createdAt
}

func (e *Entry) TTLRemaining() time.Duration {
	if e.expireAt == 0 {
		return 0
	}

	remain := time.Until(time.Unix(0, e.expireAt))
	if remain < 0 {
		return 0
	}

	return remain
}

func (e *Entry) TTLRemainingNano() int64 {
	if e.expireAt == 0 {
		return 0
	}

	remain := e.expireAt - time.Now().UnixNano()
	if remain < 0 {
		return 0
	}

	return remain
}

func (e *Entry) ExpireAt() int64 {
	return e.expireAt
}

func (e *Entry) UpdateTTL(ttl time.Duration) {
	if ttl > 0 {
		e.expireAt = time.Now().Add(ttl).UnixNano()
	} else {
		e.expireAt = 0
	}
}

func (e *Entry) Clone() *Entry {
	valueCopy := make([]byte, len(e.value))
	copy(valueCopy, e.value)

	return &Entry{
		key:       e.key,
		value:     valueCopy,
		offset:    e.offset,
		size:      e.size,
		expireAt:  e.expireAt,
		createdAt: e.createdAt,
	}
}

func (e *Entry) Score() int {
	accesses := e.accesses.Load()
	age := time.Now().UnixNano() - e.createdAt

	if age == 0 {
		return int(accesses)
	}

	frequency := float64(accesses) / (float64(age) / float64(time.Second))

	return int(frequency * 1000)
}

func (e *Entry) LFUScore() int {
	return int(e.accesses.Load())
}

func (e *Entry) LRUScore() int64 {
	return -e.lastAccess.Load()
}

func (e *Entry) SizeScore() int {
	return int(e.size)
}

func (e *Entry) Metadata() map[string]interface{} {
	return map[string]interface{}{
		"key":         e.key,
		"size":        e.size,
		"accesses":    e.accesses.Load(),
		"age_ms":      e.AgeNano() / int64(time.Millisecond),
		"ttl_ms":      e.TTLRemainingNano() / int64(time.Millisecond),
		"last_access": e.lastAccess.Load(),
		"created_at":  e.createdAt,
	}
}
