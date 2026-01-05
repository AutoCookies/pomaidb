package ttl

import (
	"context"
	"time"

	"golang.org/x/sync/singleflight"
)

const shardCount = 64

type Manager struct {
	shards [shardCount]*singleflight.Group
}

func NewManager() *Manager {
	m := &Manager{}
	for i := 0; i < shardCount; i++ {
		m.shards[i] = &singleflight.Group{}
	}
	return m
}

type LoadResult struct {
	Data []byte
	TTL  time.Duration
}

func (m *Manager) Do(
	ctx context.Context,
	key string,
	fn func(context.Context) ([]byte, time.Duration, error),
) ([]byte, bool, error) {
	if key == "" {
		return nil, false, nil
	}

	shard := m.getShard(key)

	resCh := shard.DoChan(key, func() (interface{}, error) {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		data, ttl, err := fn(ctx)
		if err != nil {
			return nil, err
		}

		return &LoadResult{Data: data, TTL: ttl}, nil
	})

	select {
	case res := <-resCh:
		if res.Err != nil {
			return nil, false, res.Err
		}
		if result, ok := res.Val.(*LoadResult); ok {
			return result.Data, res.Shared, nil
		}
		return nil, false, nil

	case <-ctx.Done():
		return nil, false, ctx.Err()
	}
}

func (m *Manager) DoWithCallback(
	ctx context.Context,
	key string,
	fn func(context.Context) ([]byte, time.Duration, error),
	onResult func(data []byte, ttl time.Duration),
) ([]byte, bool, error) {
	data, found, err := m.Do(ctx, key, fn)

	if err == nil && onResult != nil {
		onResult(data, 0)
	}

	return data, found, err
}

func (m *Manager) Forget(key string) {
	m.getShard(key).Forget(key)
}

func (m *Manager) getShard(key string) *singleflight.Group {
	h := fnv32(key)
	return m.shards[h%shardCount]
}

func fnv32(key string) uint32 {
	hash := uint32(2166136261)
	const prime32 = uint32(16777619)
	for i := 0; i < len(key); i++ {
		hash *= prime32
		hash ^= uint32(key[i])
	}
	return hash
}
