package httpadapter

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine"
	"github.com/gorilla/mux"
)

// handleHealth returns basic server status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := s.tenants.StatsAll()

	health := map[string]interface{}{
		"status":      "ok",
		"timestamp":   time.Now().Unix(),
		"total_users": len(stats),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handlePut stores a value in the cache
func (s *Server) handlePut(w http.ResponseWriter, r *http.Request) {
	tenant := tenantFromContext(r.Context())
	vars := mux.Vars(r)
	key := vars["key"]

	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	const maxSize = 10 * 1024 * 1024 // 10MB limit
	r.Body = http.MaxBytesReader(w, r.Body, maxSize)

	ttl := time.Duration(0)
	if v := r.URL.Query().Get("ttl"); v != "" {
		if secs, err := strconv.ParseInt(v, 10, 64); err == nil && secs > 0 {
			ttl = time.Duration(secs) * time.Second
		}
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body failed or too large", http.StatusBadRequest)
		return
	}

	store := s.tenants.GetStore(tenant)
	store.Put(key, body, ttl)

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

// handleGet retrieves a value
func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	tenant := tenantFromContext(r.Context())
	key := mux.Vars(r)["key"]

	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	store := s.tenants.GetStore(tenant)
	v, ok := store.Get(key)
	if !ok {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(v)
}

// handleDelete removes a key
func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	tenant := tenantFromContext(r.Context())
	key := mux.Vars(r)["key"]

	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	store := s.tenants.GetStore(tenant)
	store.Delete(key)

	w.WriteHeader(http.StatusNoContent)
}

// handleHead checks key existence and TTL
func (s *Server) handleHead(w http.ResponseWriter, r *http.Request) {
	tenant := tenantFromContext(r.Context())
	key := mux.Vars(r)["key"]

	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	store := s.tenants.GetStore(tenant)
	remain, ok := store.TTLRemaining(key)
	if !ok {
		http.NotFound(w, r)
		return
	}

	if remain > 0 {
		w.Header().Set("X-Cache-TTL-Remaining", fmt.Sprintf("%d", int64(remain.Seconds())))
	} else {
		w.Header().Set("X-Cache-TTL-Remaining", "0")
	}
	w.WriteHeader(http.StatusOK)
}

// handleStats returns cache statistics
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("userId")

	w.Header().Set("Content-Type", "application/json")

	if userID != "" {
		if st, ok := s.tenants.StatsForTenant(userID); ok {
			json.NewEncoder(w).Encode(map[string]interface{}{
				"userId": userID,
				"stats":  st,
			})
			return
		}
		http.Error(w, "user not found", http.StatusNotFound)
		return
	}

	type AggStats struct {
		TotalUsers int                     `json:"total_users"`
		PerUser    map[string]engine.Stats `json:"per_user"`
	}

	statsMap := s.tenants.StatsAll()
	out := AggStats{
		TotalUsers: len(statsMap),
		PerUser:    statsMap,
	}

	json.NewEncoder(w).Encode(out)
}

// handleIncr handles atomic increment/decrement
// Route: POST /v1/cache/{key}/incr?delta=1
func (s *Server) handleIncr(w http.ResponseWriter, r *http.Request) {
	tenant := tenantFromContext(r.Context())
	vars := mux.Vars(r)
	key := vars["key"]

	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	// Lấy delta từ query param (mặc định là 1)
	deltaStr := r.URL.Query().Get("delta")
	var delta int64 = 1
	if deltaStr != "" {
		d, err := strconv.ParseInt(deltaStr, 10, 64)
		if err != nil {
			http.Error(w, "invalid delta", http.StatusBadRequest)
			return
		}
		delta = d
	}

	store := s.tenants.GetStore(tenant)

	// Gọi xuống Engine (Hàm Incr bạn vừa thêm vào store.go)
	newVal, err := store.Incr(key, delta)
	if err != nil {
		// Trả về lỗi nếu value cũ không phải là số integer
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Trả về giá trị mới dạng JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int64{
		"value": newVal,
	})
}
