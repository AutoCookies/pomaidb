//go:build !windows

package storage

import (
	"os"
	"syscall"
)

func mmap(f *os.File, size int64) ([]byte, error) {
	return syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
}

func munmap(data []byte) error {
	return syscall.Munmap(data)
}
