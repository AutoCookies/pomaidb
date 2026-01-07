//go:build windows

package storage

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

func mmap(f *os.File, size int64) ([]byte, error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READWRITE, uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, fmt.Errorf("CreateFileMapping failed: %w", err)
	}
	defer syscall.CloseHandle(h)

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(size))
	if err != nil {
		return nil, fmt.Errorf("MapViewOfFile failed: %w", err)
	}

	var b []byte
	hdr := (*[3]uintptr)(unsafe.Pointer(&b))
	hdr[0] = addr
	hdr[1] = uintptr(size)
	hdr[2] = uintptr(size)
	return b, nil
}

func munmap(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	type sliceHeader struct {
		Data uintptr
		Len  int
		Cap  int
	}
	h := (*sliceHeader)(unsafe.Pointer(&data))
	return syscall.UnmapViewOfFile(h.Data)
}
