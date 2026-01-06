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
	slabRef    interface{}
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

func (e *Entry) TTLRemaining() time.Duration {
	if e.expireAt == 0 {
		return 0
	}
	remain := e.expireAt - time.Now().UnixNano()
	if remain < 0 {
		return 0
	}
	return time.Duration(remain)
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
		slabRef:   nil,
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

func (e *Entry) SetSlab(slab interface{}) {
	e.slabRef = slab
}

func (e *Entry) GetSlab() interface{} {
	return e.slabRef
}
