package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	SearchLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "vector_search_latency_milliseconds",
		Help:    "Search latency in milliseconds by index",
		Buckets: prometheus.ExponentialBuckets(0.1, 2.0, 20),
	}, []string{"index"})

	InsertCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "vector_insert_total",
		Help: "Total number of inserts by index",
	}, []string{"index"})

	CurrentEf = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vector_current_ef",
		Help: "Current ef_search value by index",
	}, []string{"index"})

	ArenaSize = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vector_arena_size",
		Help: "Number of vectors in arena by index",
	}, []string{"index"})

	AppliedMemoryLimit = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "sysadapt_applied_memory_limit_bytes",
		Help: "Applied Go memory limit in bytes",
	})

	OffheapBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "sysadapt_offheap_bytes",
		Help: "Off-heap bytes reported by OffHeapReporter",
	})
)

func init() {
	prometheus.MustRegister(SearchLatency)
	prometheus.MustRegister(InsertCount)
	prometheus.MustRegister(CurrentEf)
	prometheus.MustRegister(ArenaSize)
	prometheus.MustRegister(AppliedMemoryLimit)
	prometheus.MustRegister(OffheapBytes)
}

func ObserveSearchLatency(index string, ms float64) {
	SearchLatency.WithLabelValues(index).Observe(ms)
}

func IncInsert(index string) {
	InsertCount.WithLabelValues(index).Inc()
}

func SetEf(index string, ef int) {
	CurrentEf.WithLabelValues(index).Set(float64(ef))
}

func SetArenaSize(index string, size int) {
	ArenaSize.WithLabelValues(index).Set(float64(size))
}

func SetAppliedMemoryLimit(bytes int64) {
	AppliedMemoryLimit.Set(float64(bytes))
}

func SetOffheapBytes(bytes int64) {
	OffheapBytes.Set(float64(bytes))
}
