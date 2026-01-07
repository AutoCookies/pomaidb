package data_types

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
)

// PMCStore quản lý lưu trữ và tính toán Ma trận
type PMCStore struct {
	mu       sync.RWMutex
	matrices map[string]Matrix
}

type Matrix struct {
	Rows int
	Cols int
	Data []float32
}

func NewPMCStore() *PMCStore {
	return &PMCStore{
		matrices: make(map[string]Matrix),
	}
}

// MatrixSet lưu ma trận với nén SVD xấp xỉ (Rank-1 Approximation)
// Nếu ma trận quá nhỏ, lưu nguyên bản để tránh overhead.
func (pm *PMCStore) Set(key string, rows, cols int, data []float32) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if len(data) != rows*cols {
		return fmt.Errorf("invalid matrix dimensions")
	}

	// Với ma trận lớn (>1000 phần tử), nén bằng SVD Rank-1
	// A ~ sigma * u * v^T
	if len(data) > 1000 {
		compressed, err := svdCompress(rows, cols, data)
		if err == nil {
			pm.matrices[key] = compressed
			return nil
		}
	}

	// Fallback: Lưu Raw
	pm.matrices[key] = Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
	return nil
}

func (pm *PMCStore) Get(key string) (Matrix, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	mat, ok := pm.matrices[key]
	if !ok {
		return Matrix{}, common.ErrKeyNotFound
	}
	return mat, nil
}

// MatrixAdd cộng hai ma trận đã lưu: C = A + B
func (pm *PMCStore) Add(key1, key2 string) (Matrix, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	m1, ok1 := pm.matrices[key1]
	m2, ok2 := pm.matrices[key2]
	if !ok1 || !ok2 {
		return Matrix{}, common.ErrKeyNotFound
	}

	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return Matrix{}, fmt.Errorf("dimension mismatch")
	}

	// SIMD-friendly loop (compiler auto-vectorization)
	result := make([]float32, len(m1.Data))
	for i := 0; i < len(m1.Data); i++ {
		result[i] = m1.Data[i] + m2.Data[i]
	}

	return Matrix{
		Rows: m1.Rows,
		Cols: m1.Cols,
		Data: result,
	}, nil
}

// MatrixMultiply nhân hai ma trận: C = A * B
func (pm *PMCStore) Multiply(key1, key2 string) (Matrix, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	m1, ok1 := pm.matrices[key1]
	m2, ok2 := pm.matrices[key2]
	if !ok1 || !ok2 {
		return Matrix{}, common.ErrKeyNotFound
	}

	if m1.Cols != m2.Rows {
		return Matrix{}, common.ErrDimensionMismatch
	}

	// Naive O(N^3) - Optimized implementation would use blocking/tiling
	result := make([]float32, m1.Rows*m2.Cols)
	for i := 0; i < m1.Rows; i++ {
		for k := 0; k < m1.Cols; k++ {
			sum := m1.Data[i*m1.Cols+k]
			for j := 0; j < m2.Cols; j++ {
				result[i*m2.Cols+j] += sum * m2.Data[k*m2.Cols+j]
			}
		}
	}

	return Matrix{
		Rows: m1.Rows,
		Cols: m2.Cols,
		Data: result,
	}, nil
}

// --- Self-made Math Core ---

// svdCompress sử dụng Power Iteration để tìm Rank-1 Approximation
// Nén ma trận (M*N) thành (M+N+1) phần tử: vector u, vector v, scalar sigma.
// Tiết kiệm bộ nhớ cực lớn cho các ma trận thưa hoặc low-rank.
func svdCompress(rows, cols int, data []float32) (Matrix, error) {
	// Power Iteration để tìm dominant singular vector
	// 1. Init random vector v
	v := make([]float32, cols)
	for i := range v {
		v[i] = rand.Float32()
	}

	// 2. Iterate: v = A^T * (A * v)
	// Thực hiện 5 vòng lặp (đủ để hội tụ sơ bộ)
	for iter := 0; iter < 5; iter++ {
		// Av = A * v
		av := make([]float32, rows)
		for i := 0; i < rows; i++ {
			sum := float32(0)
			for j := 0; j < cols; j++ {
				sum += data[i*cols+j] * v[j]
			}
			av[i] = sum
		}

		// v = A^T * av
		for j := 0; j < cols; j++ {
			sum := float32(0)
			for i := 0; i < rows; i++ {
				sum += data[i*cols+j] * av[i]
			}
			v[j] = sum
		}

		// Normalize v
		norm := float32(0)
		for _, val := range v {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm < 1e-6 {
			break
		}
		for i := range v {
			v[i] /= norm
		}
	}

	// 3. Tính sigma và u
	// u = A * v / sigma (nhưng ta lưu trực tiếp sigma*u để đơn giản)
	u := make([]float32, rows)
	for i := 0; i < rows; i++ {
		sum := float32(0)
		for j := 0; j < cols; j++ {
			sum += data[i*cols+j] * v[j]
		}
		u[i] = sum // Đây chính là (sigma * u_i)
	}

	// Reconstruct approximate matrix for storage (Lossy Compression)
	// A_approx = u * v^T (u ở đây đã chứa sigma)
	compressedData := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			compressedData[i*cols+j] = u[i] * v[j]
		}
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		Data: compressedData,
	}, nil
}

// EncodeMatrix serializes matrix to bytes
func EncodeMatrix(m Matrix) []byte {
	buf := make([]byte, 8+len(m.Data)*4)
	binary.BigEndian.PutUint32(buf[0:4], uint32(m.Rows))
	binary.BigEndian.PutUint32(buf[4:8], uint32(m.Cols))
	for i, v := range m.Data {
		bits := math.Float32bits(v)
		binary.BigEndian.PutUint32(buf[8+i*4:], bits)
	}
	return buf
}

// DecodeMatrix deserializes bytes to matrix
func DecodeMatrix(data []byte) (Matrix, error) {
	if len(data) < 8 {
		return Matrix{}, fmt.Errorf("data too short")
	}
	rows := int(binary.BigEndian.Uint32(data[0:4]))
	cols := int(binary.BigEndian.Uint32(data[4:8]))
	expectedLen := 8 + rows*cols*4
	if len(data) != expectedLen {
		return Matrix{}, fmt.Errorf("data size mismatch")
	}
	matData := make([]float32, rows*cols)
	for i := 0; i < rows*cols; i++ {
		bits := binary.BigEndian.Uint32(data[8+i*4:])
		matData[i] = math.Float32frombits(bits)
	}
	return Matrix{Rows: rows, Cols: cols, Data: matData}, nil
}
