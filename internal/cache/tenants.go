package cache

import (
	"sync"
)

// TenantManager manages per-tenant stores.
type TenantManager struct {
	mu                sync.RWMutex
	stores            map[string]*Store
	shardCount        int
	perTenantCapacity int64
}

// NewTenantManager creates a manager that will create per-tenant stores on demand.
// shardCount is forwarded to each per-tenant store. perTenantCapacity is bytes quota per tenant (0 = unlimited).
func NewTenantManager(shardCount int, perTenantCapacity int64) *TenantManager {
	return &TenantManager{
		stores:            make(map[string]*Store),
		shardCount:        shardCount,
		perTenantCapacity: perTenantCapacity,
	}
}

// GetStore returns the Store for the given tenantID, creating it if needed.
func (tm *TenantManager) GetStore(tenantID string) *Store {
	// fast path read lock
	tm.mu.RLock()
	s, ok := tm.stores[tenantID]
	tm.mu.RUnlock()
	if ok {
		return s
	}

	// create under write lock
	tm.mu.Lock()
	defer tm.mu.Unlock()
	// double-check
	if s, ok = tm.stores[tenantID]; ok {
		return s
	}
	s = NewStoreWithOptions(tm.shardCount, tm.perTenantCapacity)
	tm.stores[tenantID] = s
	return s
}

// StatsForTenant returns Stats for a specific tenant store; returns false if tenant not found.
func (tm *TenantManager) StatsForTenant(tenantID string) (Stats, bool) {
	tm.mu.RLock()
	s, ok := tm.stores[tenantID]
	tm.mu.RUnlock()
	if !ok {
		return Stats{}, false
	}
	return s.Stats(), true
}

// ListTenants returns a slice of tenant IDs currently known.
func (tm *TenantManager) ListTenants() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	out := make([]string, 0, len(tm.stores))
	for id := range tm.stores {
		out = append(out, id)
	}
	return out
}

// StatsAll returns a map tenantID -> Stats snapshot for all known tenants.
func (tm *TenantManager) StatsAll() map[string]Stats {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	out := make(map[string]Stats, len(tm.stores))
	for id, s := range tm.stores {
		out[id] = s.Stats()
	}
	return out
}
