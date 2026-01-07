package tenants

import (
	"sync"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core"
)

type tenant struct {
	store *core.Store
	sem   chan struct{}
}

type Manager struct {
	mu                sync.RWMutex
	tenants           map[string]*tenant
	shardCount        int
	perTenantCapacity int64
	maxConcurrent     int
}

type StoreInterface interface{}

func NewManager(shardCount int, perTenantCapacity int64) *Manager {
	return NewManagerWithLimit(shardCount, perTenantCapacity, 100)
}

func NewManagerWithLimit(shardCount int, perTenantCapacity int64, maxConcurrent int) *Manager {
	if maxConcurrent <= 0 {
		maxConcurrent = 100
	}
	return &Manager{
		tenants:           make(map[string]*tenant),
		shardCount:        shardCount,
		perTenantCapacity: perTenantCapacity,
		maxConcurrent:     maxConcurrent,
	}
}

func (tm *Manager) GetStore(tenantID string) *core.Store {
	tm.mu.RLock()
	t, ok := tm.tenants[tenantID]
	tm.mu.RUnlock()

	if ok {
		return t.store
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	if t, ok = tm.tenants[tenantID]; ok {
		return t.store
	}

	store := core.NewStoreWithOptions(tm.shardCount, tm.perTenantCapacity)
	store.SetTenantID(tenantID)

	t = &tenant{
		store: store,
		sem:   make(chan struct{}, tm.maxConcurrent),
	}
	tm.tenants[tenantID] = t

	return store
}

func (tm *Manager) AcquireTenant(tenantID string, timeout time.Duration) bool {
	tm.mu.RLock()
	t, ok := tm.tenants[tenantID]
	tm.mu.RUnlock()

	if !ok {
		tm.GetStore(tenantID)
		tm.mu.RLock()
		t, ok = tm.tenants[tenantID]
		tm.mu.RUnlock()
		if !ok {
			return false
		}
	}

	if timeout <= 0 {
		t.sem <- struct{}{}
		return true
	}

	select {
	case t.sem <- struct{}{}:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (tm *Manager) ReleaseTenant(tenantID string) {
	tm.mu.RLock()
	t, ok := tm.tenants[tenantID]
	tm.mu.RUnlock()

	if !ok {
		return
	}

	select {
	case <-t.sem:
	default:
	}
}

func (tm *Manager) ListTenants() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	ids := make([]string, 0, len(tm.tenants))
	for id := range tm.tenants {
		ids = append(ids, id)
	}
	return ids
}

func (tm *Manager) RemoveTenant(tenantID string) bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, ok := tm.tenants[tenantID]; !ok {
		return false
	}

	delete(tm.tenants, tenantID)
	return true
}
