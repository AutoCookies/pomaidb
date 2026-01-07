// File: internal/adapter/tcp/client.go
package tcp

import (
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// CLIENT (OPTIMIZED)
// ============================================================================

type Client struct {
	conn      net.Conn
	bufConn   *BufferedConn
	addr      string
	timeout   time.Duration
	connected atomic.Bool
	mu        sync.Mutex

	pooled   bool
	lastUsed int64

	requestCount atomic.Uint64
	errorCount   atomic.Uint64
}

type ClientConfig struct {
	Address        string
	ConnectTimeout time.Duration
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration

	EnableKeepalive bool
	TCPNoDelay      bool
	ReadBufferSize  int
	WriteBufferSize int
}

func DefaultClientConfig() *ClientConfig {
	return &ClientConfig{
		Address:         "localhost:7600",
		ConnectTimeout:  5 * time.Second,
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    10 * time.Second,
		EnableKeepalive: true,
		TCPNoDelay:      true,
		ReadBufferSize:  256 * 1024, // 256KB
		WriteBufferSize: 256 * 1024, // 256KB
	}
}

// ============================================================================
// CONSTRUCTOR (OPTIMIZED)
// ============================================================================

func NewClient(addr string) (*Client, error) {
	config := DefaultClientConfig()
	config.Address = addr
	return NewClientWithConfig(config)
}

func NewClientWithConfig(config *ClientConfig) (*Client, error) {
	client := &Client{
		addr:    config.Address,
		timeout: config.ReadTimeout,
		pooled:  false,
	}

	if err := client.ConnectWithConfig(config); err != nil {
		return nil, err
	}

	return client, nil
}

// ============================================================================
// CONNECTION MANAGEMENT (OPTIMIZED)
// ============================================================================

// ✅ OPTIMIZED: Better TCP tuning
func (c *Client) ConnectWithConfig(config *ClientConfig) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected.Load() {
		return nil
	}

	conn, err := net.DialTimeout("tcp", c.addr, config.ConnectTimeout)
	if err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("failed to connect:  %w", err)
	}

	// ✅ OPTIMIZED: Configure TCP connection
	if tcpConn, ok := conn.(*net.TCPConn); ok {
		if config.EnableKeepalive {
			tcpConn.SetKeepAlive(true)
			tcpConn.SetKeepAlivePeriod(time.Minute)
		}
		if config.TCPNoDelay {
			tcpConn.SetNoDelay(true)
		}
		if config.ReadBufferSize > 0 {
			tcpConn.SetReadBuffer(config.ReadBufferSize)
		}
		if config.WriteBufferSize > 0 {
			tcpConn.SetWriteBuffer(config.WriteBufferSize)
		}
	}

	c.conn = conn
	c.bufConn = NewBufferedConn(conn)
	c.connected.Store(true)
	c.lastUsed = time.Now().UnixNano()

	return nil
}

func (c *Client) Connect() error {
	return c.ConnectWithConfig(DefaultClientConfig())
}

func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected.Load() {
		return nil
	}

	c.connected.Store(false)
	return c.conn.Close()
}

// ✅ OPTIMIZED: Lock-free check
func (c *Client) IsConnected() bool {
	return c.connected.Load()
}

// ✅ NEW: Get connection stats
func (c *Client) Stats() map[string]interface{} {
	return map[string]interface{}{
		"requests":  c.requestCount.Load(),
		"errors":    c.errorCount.Load(),
		"connected": c.connected.Load(),
	}
}

// ============================================================================
// OPTIMIZED BASIC OPERATIONS
// ============================================================================

// ✅ OPTIMIZED: Single lock-free fast path
func (c *Client) Set(key string, value []byte) error {
	if !c.connected.Load() {
		return fmt.Errorf("not connected")
	}

	c.requestCount.Add(1)

	// Set deadlines
	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	// Send request
	if err := c.bufConn.WritePacket(OpSet, key, value); err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("write error: %w", err)
	}

	// Read response
	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return nil
}

func (c *Client) Get(key string) ([]byte, error) {
	if !c.connected.Load() {
		return nil, fmt.Errorf("not connected")
	}

	c.requestCount.Add(1)

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	if err := c.bufConn.WritePacket(OpGet, key, nil); err != nil {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode == StatusKeyNotFound {
		return nil, fmt.Errorf("key not found")
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return resp.Value, nil
}

func (c *Client) Delete(key string) error {
	if !c.connected.Load() {
		return fmt.Errorf("not connected")
	}

	c.requestCount.Add(1)

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	if err := c.bufConn.WritePacket(OpDel, key, nil); err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return nil
}

func (c *Client) Exists(key string) (bool, error) {
	if !c.connected.Load() {
		return false, fmt.Errorf("not connected")
	}

	c.requestCount.Add(1)

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	if err := c.bufConn.WritePacket(OpExists, key, nil); err != nil {
		c.errorCount.Add(1)
		return false, fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return false, fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return false, fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return string(resp.Value) == "1", nil
}

func (c *Client) Incr(key string, delta int64) (int64, error) {
	if !c.connected.Load() {
		return 0, fmt.Errorf("not connected")
	}

	c.requestCount.Add(1)

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	deltaBytes := []byte(fmt.Sprintf("%d", delta))
	if err := c.bufConn.WritePacket(OpIncr, key, deltaBytes); err != nil {
		c.errorCount.Add(1)
		return 0, fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return 0, fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return 0, fmt.Errorf("server error: %s", string(resp.Value))
	}

	var result int64
	fmt.Sscanf(string(resp.Value), "%d", &result)

	c.lastUsed = time.Now().UnixNano()
	return result, nil
}

func (c *Client) Pipeline(cmds []Packet) ([]Packet, error) {
	if !c.connected.Load() {
		return nil, fmt.Errorf("not connected")
	}

	if len(cmds) == 0 {
		return nil, nil
	}

	c.requestCount.Add(uint64(len(cmds)))

	timeout := c.timeout
	if len(cmds) > 100 {
		timeout = time.Duration(len(cmds)/100+1) * c.timeout
	}

	deadline := time.Now().Add(timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	for i := range cmds {
		cmd := &cmds[i]
		if err := c.bufConn.WritePacket(cmd.Opcode, cmd.Key, cmd.Value); err != nil {
			c.errorCount.Add(1)
			return nil, fmt.Errorf("pipeline write error at %d: %w", i, err)
		}
	}

	if err := c.bufConn.writer.Flush(); err != nil {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("flush error: %w", err)
	}

	responses := make([]Packet, len(cmds))
	for i := 0; i < len(cmds); i++ {
		resp, err := c.bufConn.ReadPacket()
		if err != nil {
			c.errorCount.Add(1)
			return nil, fmt.Errorf("pipeline read error at %d:  %w", i, err)
		}
		responses[i] = resp
	}

	c.lastUsed = time.Now().UnixNano()
	return responses, nil
}

func (c *Client) PipelineFast(cmds []Packet, responses []Packet) error {
	if !c.connected.Load() {
		return fmt.Errorf("not connected")
	}

	if len(cmds) == 0 {
		return nil
	}

	if len(responses) < len(cmds) {
		return fmt.Errorf("response buffer too small")
	}

	c.requestCount.Add(uint64(len(cmds)))

	timeout := c.timeout
	if len(cmds) > 100 {
		timeout = time.Duration(len(cmds)/100+1) * c.timeout
	}

	deadline := time.Now().Add(timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	// Write all
	for i := range cmds {
		cmd := &cmds[i]
		if err := c.bufConn.WritePacket(cmd.Opcode, cmd.Key, cmd.Value); err != nil {
			c.errorCount.Add(1)
			return fmt.Errorf("write error at %d: %w", i, err)
		}
	}

	// Flush once
	if err := c.bufConn.writer.Flush(); err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("flush error: %w", err)
	}

	// Read all into pre-allocated buffer
	for i := 0; i < len(cmds); i++ {
		resp, err := c.bufConn.ReadPacket()
		if err != nil {
			c.errorCount.Add(1)
			return fmt.Errorf("read error at %d: %w", i, err)
		}
		responses[i] = resp
	}

	c.lastUsed = time.Now().UnixNano()
	return nil
}

// ============================================================================
// OPTIMIZED BATCH OPERATIONS
// ============================================================================

// ✅ OPTIMIZED: MGet with better buffer management
func (c *Client) MGet(keys []string) (map[string][]byte, error) {
	if !c.connected.Load() {
		return nil, fmt.Errorf("not connected")
	}

	if len(keys) == 0 {
		return nil, nil
	}

	c.requestCount.Add(1)

	// Encode keys (optimized)
	totalSize := 2
	for _, key := range keys {
		totalSize += 2 + len(key)
	}

	buf := make([]byte, totalSize)
	offset := 0

	count := len(keys)
	buf[offset] = byte(count >> 8)
	buf[offset+1] = byte(count)
	offset += 2

	for _, key := range keys {
		keyLen := len(key)
		buf[offset] = byte(keyLen >> 8)
		buf[offset+1] = byte(keyLen)
		offset += 2
		copy(buf[offset:], key)
		offset += keyLen
	}

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	if err := c.bufConn.WritePacket(OpMGet, "", buf); err != nil {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return nil, fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return decodeMGetResponse(resp.Value)
}

// ✅ OPTIMIZED: MSet with better encoding
func (c *Client) MSet(items map[string][]byte) error {
	if !c.connected.Load() {
		return fmt.Errorf("not connected")
	}

	if len(items) == 0 {
		return nil
	}

	c.requestCount.Add(1)

	// Encode items (optimized)
	totalSize := 2
	for key, val := range items {
		totalSize += 2 + len(key) + 4 + len(val)
	}

	buf := make([]byte, totalSize)
	offset := 0

	count := len(items)
	buf[offset] = byte(count >> 8)
	buf[offset+1] = byte(count)
	offset += 2

	for key, val := range items {
		keyLen := len(key)
		buf[offset] = byte(keyLen >> 8)
		buf[offset+1] = byte(keyLen)
		offset += 2
		copy(buf[offset:], key)
		offset += keyLen

		valLen := len(val)
		buf[offset] = byte(valLen >> 24)
		buf[offset+1] = byte(valLen >> 16)
		buf[offset+2] = byte(valLen >> 8)
		buf[offset+3] = byte(valLen)
		offset += 4
		copy(buf[offset:], val)
		offset += valLen
	}

	deadline := time.Now().Add(c.timeout)
	c.conn.SetWriteDeadline(deadline)
	c.conn.SetReadDeadline(deadline)

	if err := c.bufConn.WritePacket(OpMSet, "", buf); err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("write error: %w", err)
	}

	resp, err := c.bufConn.ReadPacket()
	if err != nil {
		c.errorCount.Add(1)
		return fmt.Errorf("read error: %w", err)
	}

	if resp.Opcode != StatusOK {
		c.errorCount.Add(1)
		return fmt.Errorf("server error: %s", string(resp.Value))
	}

	c.lastUsed = time.Now().UnixNano()
	return nil
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

func decodeMGetResponse(data []byte) (map[string][]byte, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("invalid response")
	}

	count := int(data[0])<<8 | int(data[1])
	result := make(map[string][]byte, count)
	offset := 2

	for i := 0; i < count; i++ {
		if offset+2 > len(data) {
			return nil, fmt.Errorf("truncated response")
		}
		keyLen := int(data[offset])<<8 | int(data[offset+1])
		offset += 2

		if offset+keyLen > len(data) {
			return nil, fmt.Errorf("truncated response")
		}
		key := string(data[offset : offset+keyLen])
		offset += keyLen

		if offset+4 > len(data) {
			return nil, fmt.Errorf("truncated response")
		}
		valLen := int(data[offset])<<24 | int(data[offset+1])<<16 | int(data[offset+2])<<8 | int(data[offset+3])
		offset += 4

		if offset+valLen > len(data) {
			return nil, fmt.Errorf("truncated response")
		}
		value := make([]byte, valLen)
		copy(value, data[offset:offset+valLen])
		offset += valLen

		result[key] = value
	}

	return result, nil
}
