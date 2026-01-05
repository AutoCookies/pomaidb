package storage

import (
	"errors"
	"os"
	"sync"
)

const (
	PageSize    = 4096
	GranuleSize = 64
	RamMask     = uint64(1 << 63)
)

type VirtualPtr uint64

type VirtualStore struct {
	mu          sync.RWMutex
	file        *os.File
	mmapData    []byte
	mapSize     int64
	writeOffset int64

	ramPool [][]byte
	ramFree []int
}

func NewVirtualStore(filePath string, initialSize int64) (*VirtualStore, error) {
	f, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}

	info, _ := f.Stat()
	currentSize := info.Size()

	if currentSize < initialSize {
		if err := f.Truncate(initialSize); err != nil {
			f.Close()
			return nil, err
		}
		currentSize = initialSize
	}

	data, err := mmap(f, currentSize)
	if err != nil {
		f.Close()
		return nil, err
	}

	return &VirtualStore{
		file:        f,
		mmapData:    data,
		mapSize:     currentSize,
		writeOffset: currentSize,
		ramPool:     make([][]byte, 0, 1024),
		ramFree:     make([]int, 0, 1024),
	}, nil
}

func (v *VirtualStore) Alloc(data []byte, forceRam bool) (VirtualPtr, error) {
	if len(data) > GranuleSize {
		return 0, errors.New("data exceeds granule size")
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	if forceRam || len(v.ramFree) > 0 {
		var idx int
		if len(v.ramFree) > 0 {
			idx = v.ramFree[len(v.ramFree)-1]
			v.ramFree = v.ramFree[:len(v.ramFree)-1]
			copy(v.ramPool[idx], data)
		} else {
			buf := make([]byte, GranuleSize)
			copy(buf, data)
			v.ramPool = append(v.ramPool, buf)
			idx = len(v.ramPool) - 1
		}
		return VirtualPtr(uint64(idx) | RamMask), nil
	}

	if v.writeOffset+GranuleSize > v.mapSize {
		return 0, errors.New("virtual memory full")
	}

	offset := v.writeOffset
	start := int(offset)
	end := start + GranuleSize
	copy(v.mmapData[start:end], data)
	v.writeOffset += GranuleSize

	return VirtualPtr(uint64(offset)), nil
}

func (v *VirtualStore) Get(ptr VirtualPtr) ([]byte, bool) {
	val := uint64(ptr)
	isRam := (val & RamMask) != 0
	idx := val &^ RamMask

	v.mu.RLock()
	defer v.mu.RUnlock()

	if isRam {
		i := int(idx)
		if i >= len(v.ramPool) {
			return nil, false
		}
		return v.ramPool[i], true
	}

	start := int64(idx)
	if start+GranuleSize > v.mapSize {
		return nil, false
	}
	return v.mmapData[int(start) : int(start)+GranuleSize], true
}

func (v *VirtualStore) Demote(ptr VirtualPtr) (VirtualPtr, bool) {
	val := uint64(ptr)
	if (val & RamMask) == 0 {
		return ptr, true
	}

	idx := int(val &^ RamMask)

	v.mu.Lock()
	defer v.mu.Unlock()

	if idx >= len(v.ramPool) {
		return 0, false
	}

	data := v.ramPool[idx]
	if v.writeOffset+GranuleSize > v.mapSize {
		return 0, false
	}

	diskOffset := v.writeOffset
	start := int(diskOffset)
	end := start + GranuleSize
	copy(v.mmapData[start:end], data)
	v.writeOffset += GranuleSize

	v.ramFree = append(v.ramFree, idx)

	return VirtualPtr(uint64(diskOffset)), true
}

func (v *VirtualStore) Close() {
	if v.mmapData != nil {
		munmap(v.mmapData)
	}
	if v.file != nil {
		v.file.Close()
	}
}
