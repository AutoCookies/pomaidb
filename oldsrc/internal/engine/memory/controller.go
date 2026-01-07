package memory

import (
	"sync/atomic"
)

type Controller struct {
	capacity int64
	used     int64
	tenants  TenantManagerInterface
}

type TenantManagerInterface interface {
	ListTenants() []string
	GetStore(tenantID string) StoreInterface
}

type StoreInterface interface {
	ForceEvictBytes(target int64) int64
}

func NewController(tenants TenantManagerInterface, capacityBytes int64) *Controller {
	return &Controller{
		capacity: capacityBytes,
		used:     0,
		tenants:  tenants,
	}
}

func (mc *Controller) Used() int64 {
	return atomic.LoadInt64(&mc.used)
}

func (mc *Controller) Capacity() int64 {
	return mc.capacity
}

func (mc *Controller) UsagePercent() float64 {
	cap := mc.capacity
	if cap == 0 {
		return 0
	}
	return float64(atomic.LoadInt64(&mc.used)) / float64(cap) * 100
}

func (mc *Controller) Reserve(n int64) bool {
	if n <= 0 {
		return true
	}

	if mc.capacity == 0 {
		atomic.AddInt64(&mc.used, n)
		return true
	}

	if atomic.AddInt64(&mc.used, n) <= mc.capacity {
		return true
	}

	atomic.AddInt64(&mc.used, -n)

	if mc.evict(n) >= n {
		if atomic.AddInt64(&mc.used, n) <= mc.capacity {
			return true
		}
		atomic.AddInt64(&mc.used, -n)
	}

	return false
}

func (mc *Controller) Release(n int64) {
	if n <= 0 {
		return
	}
	val := atomic.AddInt64(&mc.used, -n)
	if val < 0 {
		atomic.CompareAndSwapInt64(&mc.used, val, 0)
	}
}

func (mc *Controller) evict(target int64) int64 {
	if target <= 0 || mc.tenants == nil {
		return 0
	}

	tenantIDs := mc.tenants.ListTenants()
	count := len(tenantIDs)
	if count == 0 {
		return 0
	}

	var totalFreed int64
	perStoreTarget := target / int64(count)
	if perStoreTarget == 0 {
		perStoreTarget = target
	}

	for _, tenantID := range tenantIDs {
		store := mc.tenants.GetStore(tenantID)
		if store != nil {
			freed := store.ForceEvictBytes(perStoreTarget)
			totalFreed += freed
			target -= freed
			if target <= 0 {
				break
			}
		}
	}

	return totalFreed
}
