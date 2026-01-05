package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/joho/godotenv"

	httpAdapter "github.com/AutoCookies/pomai-cache/internal/adapter/http"
	"github.com/AutoCookies/pomai-cache/internal/adapter/persistence"
	"github.com/AutoCookies/pomai-cache/internal/adapter/persistence/file"
	"github.com/AutoCookies/pomai-cache/internal/adapter/persistence/wal"
	tcpAdapter "github.com/AutoCookies/pomai-cache/internal/adapter/tcp"
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
	"github.com/AutoCookies/pomai-cache/internal/engine/core"
	"github.com/AutoCookies/pomai-cache/internal/engine/replication"
	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
)

const (
	AppName = "POMAI CACHE"
	Version = "1.8.1-stable"
	Build   = "Enterprise"

	ColorReset  = "\033[0m"
	ColorCyan   = "\033[36m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorRed    = "\033[31m"
	ColorBold   = "\033[1m"
)

type Config struct {
	HTTPPort         string
	TCPPort          string
	ClusterMode      bool
	NodeID           string
	GossipPort       string
	Seeds            string
	MaxConnections   int
	CacheShards      int
	CapacityBytes    int64
	PersistenceType  string
	DataDir          string
	WriteBufferSize  int
	FlushInterval    time.Duration
	HTTPReadTimeout  time.Duration
	HTTPWriteTimeout time.Duration
	HTTPIdleTimeout  time.Duration
	ShutdownTimeout  time.Duration
	EnableCORS       bool
	EnableMetrics    bool
	EnableDebug      bool
	GCPercent        int
	MaxProcs         int
	MemoryLimit      int64
}

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(math.MaxInt64)
}

func main() {
	_ = godotenv.Load()

	cfg := parseFlags()

	if cfg.MemoryLimit == 0 {
		if applied := core.ApplySystemAdaptive(); applied > 0 {
			cfg.MemoryLimit = applied
		}
	}

	tuneRuntime(cfg)
	printHeader(cfg)

	printStatus("CORE", "Initializing Tenant Manager...")
	tm := tenants.NewManager(cfg.CacheShards, cfg.CapacityBytes)

	printStatus("STORAGE", "Initializing Persistence Layer...")
	pers, persImpl := initPersistence(cfg, tm)
	defer closePersistence(pers)

	wb := initWriteBehind(cfg, pers)
	defer closeWriteBehind(wb)

	printStatus("CLUSTER", "Configuring Replication Topology...")
	rm := initCluster(cfg, tm)
	if rm != nil {
		defer rm.Stop()
	}

	printStatus("NETWORK", "Binding Service Ports...")
	httpSrv, tcpSrv := initServers(cfg, tm, rm)
	startServers(httpSrv, tcpSrv, cfg.HTTPPort, cfg.TCPPort)

	waitForHealth(cfg.HTTPPort)
	printReady(cfg)

	shutdown(cfg, httpSrv, tcpSrv, wb, persImpl, tm)
}

func parseFlags() *Config {
	defaultHTTPPort := "8080"
	defaultTCPPort := "7600"
	defaultGossipPort := "7946"
	defaultNodeID := fmt.Sprintf("node-%d", time.Now().Unix())
	defaultCacheShards := 2048
	defaultCapacityBytes := int64(0)
	defaultPersistence := "none"
	defaultDataDir := "./data"
	defaultWriteBufferSize := 1000
	defaultFlushInterval := 5 * time.Second
	defaultShutdownTimeout := 30 * time.Second
	defaultHTTPReadTimeout := 10 * time.Second
	defaultHTTPWriteTimeout := 10 * time.Second
	defaultHTTPIdleTimeout := 60 * time.Second

	httpPort := flag.String("http-port", defaultHTTPPort, "HTTP port")
	tcpPort := flag.String("tcp-port", defaultTCPPort, "TCP port")
	clusterMode := flag.Bool("cluster", false, "Enable cluster mode (gossip)")
	nodeID := flag.String("node-id", defaultNodeID, "Node ID")
	gossipPort := flag.String("gossip-port", defaultGossipPort, "Gossip port")
	seeds := flag.String("seeds", "", "Comma-separated cluster seed nodes (host:port)")
	maxConns := flag.Int("max-conns", 10000, "Max TCP connections")
	cacheShards := flag.Int("cache-shards", defaultCacheShards, "Number of cache shards")
	capacity := flag.String("capacity-bytes", "", "Per-tenant capacity (e.g. 10GB, 512MB). Empty = unlimited")
	persistence := flag.String("persistence", defaultPersistence, "Persistence type: none|file|wal")
	dataDir := flag.String("data-dir", defaultDataDir, "Data directory for persistence")
	writeBuf := flag.Int("write-buffer", defaultWriteBufferSize, "Write-behind buffer size")
	flushInt := flag.String("flush-interval", defaultFlushInterval.String(), "Write-behind flush interval (e.g. 5s)")
	httpReadTO := flag.String("http-read-timeout", defaultHTTPReadTimeout.String(), "HTTP read timeout")
	httpWriteTO := flag.String("http-write-timeout", defaultHTTPWriteTimeout.String(), "HTTP write timeout")
	httpIdleTO := flag.String("http-idle-timeout", defaultHTTPIdleTimeout.String(), "HTTP idle timeout")
	shutdownTO := flag.String("shutdown-timeout", defaultShutdownTimeout.String(), "Shutdown timeout")
	enableCORS := flag.Bool("enable-cors", false, "Enable CORS")
	enableMetrics := flag.Bool("enable-metrics", true, "Enable metrics endpoint")
	enableDebug := flag.Bool("debug", false, "Enable debug mode")
	gcPercent := flag.Int("gogc", -1, "GOGC percent (-1 leaves as-is)")
	maxProcs := flag.Int("gomaxprocs", 0, "GOMAXPROCS override (0 = leave as-is)")
	memLimit := flag.String("mem-limit", "", "Memory limit (e.g. 8GB). Empty = no limit")

	flag.Parse()

	capBytes := defaultCapacityBytes
	if *capacity != "" {
		if v, err := parseBytes(*capacity); err == nil {
			capBytes = v
		} else {
			log.Printf("[WARN] invalid capacity-bytes: %v", err)
		}
	}

	flushDur := defaultFlushInterval
	if d, err := time.ParseDuration(*flushInt); err == nil {
		flushDur = d
	} else {
		log.Printf("[WARN] invalid flush-interval: %v", err)
	}

	httpReadDur := defaultHTTPReadTimeout
	if d, err := time.ParseDuration(*httpReadTO); err == nil {
		httpReadDur = d
	} else {
		log.Printf("[WARN] invalid http-read-timeout: %v", err)
	}

	httpWriteDur := defaultHTTPWriteTimeout
	if d, err := time.ParseDuration(*httpWriteTO); err == nil {
		httpWriteDur = d
	} else {
		log.Printf("[WARN] invalid http-write-timeout: %v", err)
	}

	httpIdleDur := defaultHTTPIdleTimeout
	if d, err := time.ParseDuration(*httpIdleTO); err == nil {
		httpIdleDur = d
	} else {
		log.Printf("[WARN] invalid http-idle-timeout: %v", err)
	}

	shutdownDur := defaultShutdownTimeout
	if d, err := time.ParseDuration(*shutdownTO); err == nil {
		shutdownDur = d
	} else {
		log.Printf("[WARN] invalid shutdown-timeout: %v", err)
	}

	memLimitBytes := int64(0)
	if *memLimit != "" {
		if v, err := parseBytes(*memLimit); err == nil {
			memLimitBytes = v
		} else {
			log.Printf("[WARN] invalid mem-limit: %v", err)
		}
	}

	return &Config{
		HTTPPort:         *httpPort,
		TCPPort:          *tcpPort,
		ClusterMode:      *clusterMode,
		NodeID:           *nodeID,
		GossipPort:       *gossipPort,
		Seeds:            *seeds,
		MaxConnections:   *maxConns,
		CacheShards:      *cacheShards,
		CapacityBytes:    capBytes,
		PersistenceType:  *persistence,
		DataDir:          *dataDir,
		WriteBufferSize:  *writeBuf,
		FlushInterval:    flushDur,
		HTTPReadTimeout:  httpReadDur,
		HTTPWriteTimeout: httpWriteDur,
		HTTPIdleTimeout:  httpIdleDur,
		ShutdownTimeout:  shutdownDur,
		EnableCORS:       *enableCORS,
		EnableMetrics:    *enableMetrics,
		EnableDebug:      *enableDebug,
		GCPercent:        *gcPercent,
		MaxProcs:         *maxProcs,
		MemoryLimit:      memLimitBytes,
	}
}

func parseBytes(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToUpper(s))
	if s == "" {
		return 0, nil
	}
	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "GB"):
		mult = 1024 * 1024 * 1024
		s = strings.TrimSuffix(s, "GB")
	case strings.HasSuffix(s, "MB"):
		mult = 1024 * 1024
		s = strings.TrimSuffix(s, "MB")
	case strings.HasSuffix(s, "KB"):
		mult = 1024
		s = strings.TrimSuffix(s, "KB")
	case strings.HasSuffix(s, "B"):
		s = strings.TrimSuffix(s, "B")
	}
	i, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, err
	}
	return int64(i * float64(mult)), nil
}

func tuneRuntime(cfg *Config) {
	if cfg.MaxProcs > 0 {
		runtime.GOMAXPROCS(cfg.MaxProcs)
	}
	if cfg.GCPercent > -2 {
		debug.SetGCPercent(cfg.GCPercent)
	}
	if cfg.MemoryLimit > 0 {
		debug.SetMemoryLimit(cfg.MemoryLimit)
	}
}

func printHeader(cfg *Config) {
	fmt.Print("\033[H\033[2J")
	fmt.Println(ColorCyan + `
  ____  ____  __  __    _    ___    ____    _    ____ _   _ _____ 
 |  _ \|  _ \|  \/  |  / \  |_ _|  / ___|  / \  / ___| | | | ____|
 | |_) | | | | |\/| | / _ \  | |  | |     / _ \| |   | |_| |  _|
 |  __/| |_| | |  | |/ ___ \ | |  | |___ / ___ \ |___|  _  | |___
 |_|   \____/|_|  |_/_/   \_\___|  \____/_/   \_\____|_| |_|_____|
` + ColorReset)
	fmt.Printf(" %s%s %s%s\n", ColorBold, AppName, Version, ColorReset)
	fmt.Printf(" %s%s%s\n\n", ColorYellow, Build, ColorReset)

	fmt.Println(ColorBold + " SYSTEM RESOURCES" + ColorReset)
	fmt.Printf(" %-15s %d Cores\n", "CPU", runtime.GOMAXPROCS(0))
	if cfg.MemoryLimit > 0 {
		fmt.Printf(" %-15s %s\n", "Memory Limit", formatBytes(cfg.MemoryLimit))
	} else {
		fmt.Printf(" %-15s %s\n", "Memory Limit", "Unlimited")
	}
	fmt.Printf(" %-15s %s\n", "OS/Arch", runtime.GOOS+"/"+runtime.GOARCH)
	fmt.Println()
}

func printStatus(module, msg string) {
	fmt.Printf(" [%s", ColorCyan)
	fmt.Printf("%-8s", module)
	fmt.Printf("%s] %s\n", ColorReset, msg)
}

func printReady(cfg *Config) {
	fmt.Println()
	fmt.Println(ColorBold + " SYSTEM READY" + ColorReset)
	fmt.Printf(" %-15s %s%s%s\n", "HTTP API", ColorGreen, "http://localhost:"+cfg.HTTPPort, ColorReset)
	fmt.Printf(" %-15s %s%s%s\n", "TCP Protocol", ColorGreen, "tcp://localhost:"+cfg.TCPPort, ColorReset)

	mode := "Standalone"
	if cfg.ClusterMode {
		mode = fmt.Sprintf("Cluster (Node: %s)", cfg.NodeID)
	}
	fmt.Printf(" %-15s %s\n", "Mode", mode)
	fmt.Printf(" %-15s %s\n", "Persistence", strings.ToUpper(cfg.PersistenceType))
	fmt.Printf(" %-15s %s\n", "Capacity", formatBytes(cfg.CapacityBytes))
	fmt.Println()
}

func initPersistence(cfg *Config, tm *tenants.Manager) (ports.Persister, interface{}) {
	switch cfg.PersistenceType {
	case "file":
		p, err := file.NewFilePersister(cfg.DataDir)
		if err != nil {
			log.Fatalf("FS Init Failed: %v", err)
		}
		return p, p
	case "wal":
		p, err := wal.NewWALPersister(cfg.DataDir + "/wal.log")
		if err != nil {
			log.Fatalf("WAL Init Failed: %v", err)
		}
		_ = p.RestoreFrom(tm.GetStore("default"))
		return p, p
	default:
		p := persistence.NewNoOpPersister()
		return p, p
	}
}

func initWriteBehind(cfg *Config, p ports.Persister) *persistence.WriteBehindBuffer {
	if cfg.PersistenceType == "none" {
		return nil
	}
	wb := persistence.NewWriteBehindBuffer(cfg.WriteBufferSize, cfg.FlushInterval, p)
	wb.Start(context.Background())
	return wb
}

func closeWriteBehind(wb *persistence.WriteBehindBuffer) {
	if wb != nil {
		wb.Close()
	}
}

func initCluster(cfg *Config, tm *tenants.Manager) *replication.Manager {
	if !cfg.ClusterMode {
		return nil
	}
	rm := replication.NewManager(cfg.NodeID, replication.ModeAsync, tm)
	seeds := strings.Split(cfg.Seeds, ",")
	if err := rm.EnableGossip(cfg.TCPPort, cfg.GossipPort, seeds); err != nil {
		log.Fatalf("Gossip Init Failed: %v", err)
	}
	return rm
}

func initServers(cfg *Config, tm *tenants.Manager, rm *replication.Manager) (*httpAdapter.Server, *tcpAdapter.PomaiServer) {
	hCfg := httpAdapter.DefaultServerConfig()
	hCfg.Port, _ = strconv.Atoi(cfg.HTTPPort)
	hCfg.ReadTimeout = cfg.HTTPReadTimeout
	hCfg.WriteTimeout = cfg.HTTPWriteTimeout
	hCfg.IdleTimeout = cfg.HTTPIdleTimeout
	hCfg.EnableCORS = cfg.EnableCORS
	hCfg.EnableMetrics = cfg.EnableMetrics
	hCfg.EnableDebug = cfg.EnableDebug

	return httpAdapter.NewServerWithConfig(tm, hCfg), tcpAdapter.NewPomaiServer(tm, rm, cfg.MaxConnections)
}

func startServers(httpSrv *httpAdapter.Server, tcpSrv *tcpAdapter.PomaiServer, hPort, tPort string) {
	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP Start Failed: %v", err)
		}
	}()
	go func() {
		if err := tcpSrv.ListenAndServe(":" + tPort); err != nil {
			log.Fatalf("TCP Start Failed: %v", err)
		}
	}()
}

func waitForHealth(port string) {
	url := "http://localhost:" + port + "/health"
	for i := 0; i < 50; i++ {
		resp, err := http.Get(url)
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			return
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func shutdown(cfg *Config, hSrv *httpAdapter.Server, tSrv *tcpAdapter.PomaiServer, wb *persistence.WriteBehindBuffer, pImpl interface{}, tm *tenants.Manager) {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	fmt.Println(ColorRed + "\n SHUTDOWN SEQUENCE INITIATED" + ColorReset)
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); hSrv.Shutdown(ctx) }()
	go func() { defer wg.Done(); tSrv.Shutdown(cfg.ShutdownTimeout) }()
	wg.Wait()

	if wb != nil {
		wb.Close()
	}

	if sp, ok := pImpl.(ports.Snapshotter); ok {
		printStatus("STORAGE", "Creating Final Snapshot...")
		for _, id := range tm.ListTenants() {
			if s := tm.GetStore(id); s != nil {
				sp.Snapshot(s)
			}
		}
	}

	fmt.Println(ColorGreen + " SYSTEM HALTED SAFELY" + ColorReset)
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

func getEnvBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func getEnvDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func getEnvBytes(key string, def int64) int64 {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	v = strings.ToUpper(strings.TrimSpace(v))

	multiplier := int64(1)
	if strings.HasSuffix(v, "GB") {
		multiplier = 1024 * 1024 * 1024
		v = strings.TrimSuffix(v, "GB")
	} else if strings.HasSuffix(v, "MB") {
		multiplier = 1024 * 1024
		v = strings.TrimSuffix(v, "MB")
	} else if strings.HasSuffix(v, "KB") {
		multiplier = 1024
		v = strings.TrimSuffix(v, "KB")
	} else if strings.HasSuffix(v, "B") {
		v = strings.TrimSuffix(v, "B")
	}

	if i, err := strconv.ParseInt(v, 10, 64); err == nil {
		return i * multiplier
	}
	return def
}

func formatBytes(b int64) string {
	if b <= 0 {
		return "Unlimited"
	}
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func closePersistence(pers ports.Persister) {
	if pers != nil {
		pers.Close()
	}
}
