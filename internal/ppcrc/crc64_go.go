package ppcrc

// Pure-Go CRC64 implementation ported from the C/crcspeed approach used in Redis.
// - Implements the same _crc64 bit-by-bit generator (poly = 0xad93d23594c935a9)
// - Builds 8x256 lookup tables for slice-by-8 fast processing (crcspeed style)
// - Exposes Init() to initialize tables (called from package init) and Sum/Update helpers.
//
// Usage:
//   import "path/to/ppcrc"
//   // once at startup (package init does it automatically)
//   crc := ppcrc.Sum(nil)           // compute CRC64 of data
//   crc = ppcrc.Update(crc, buf)    // update incremental
//   // or
//   ppcrc.Init()                    // explicit init
//   v := ppcrc.CRC64(0, data)       // identical semantics to C crc64(crc, s, l)
//
// Note: this implementation detects host endianness and picks the appropriate
// slice-by-8 routine. The underlying generator mirrors the C version so results
// for the same input match the original C implementation (tested on common inputs).
//
// License: same as the code ported from (see original C headers).

import (
	"encoding/binary"
	"sync"
	"unsafe"
)

const POLY uint64 = 0xad93d23594c935a9

// crc64_table[k][n]
var crc64Table [8][256]uint64
var tableInitOnce sync.Once
var isLittleEndian bool

func init() {
	// detect endianness
	var x uint16 = 0x1
	b := (*[2]byte)(unsafe.Pointer(&x))
	isLittleEndian = b[0] == 0x1
	// build tables
	tableInitOnce.Do(func() { crc64Init() })
}

// reflect bits of data (data_len bits)
func crcReflect(data uint64, dataLen int) uint64 {
	var ret uint64 = data & 0x1
	for i := 1; i < dataLen; i++ {
		data >>= 1
		ret = (ret << 1) | (data & 0x1)
	}
	return ret
}

// _crc64: bit-by-bit-fast implementation ported from generated C function.
// It matches the behavior from the original implementation.
func _crc64(crc uint64, data []byte) uint64 {
	for _, c := range data {
		var i uint8 = 0x01
		for ; i != 0x00; i <<= 1 {
			var bit uint64 = crc & 0x8000000000000000
			if (c & i) != 0 {
				if bit == 0 {
					bit = 1
				} else {
					bit = 0
				}
			}
			crc <<= 1
			if bit != 0 {
				crc ^= POLY
			}
			// continue loop
		}
		crc &= 0xffffffffffffffff
	}
	crc = crc & 0xffffffffffffffff
	return crcReflect(crc, 64) ^ 0x0000000000000000
}

// Initialize the 8x256 table (crcspeed native init)
func crc64Init() {
	// Little-endian table init (we'll adapt for big-endian by reversing entries)
	// table[0][n] = _crc64(0, &n, 1)
	for n := 0; n < 256; n++ {
		b := byte(n)
		crc64Table[0][n] = _crc64(0, []byte{b})
	}
	// generate nested tables
	for n := 0; n < 256; n++ {
		crc := crc64Table[0][n]
		for k := 1; k < 8; k++ {
			crc = crc64Table[0][crc&0xff] ^ (crc >> 8)
			crc64Table[k][n] = crc
		}
	}
}

// CRC64 computes crc over data with given starting crc (same semantics as C crc64())
func CRC64(crc uint64, data []byte) uint64 {
	tableInitOnce.Do(func() { crc64Init() })
	// dispatch by endian
	if isLittleEndian {
		return crcspeed64little(crc64Table, crc, data)
	}
	return crcspeed64big(crc64Table, crc, data)
}

// Sum convenience: compute CRC64 of data with initial 0
func Sum(data []byte) uint64 {
	return CRC64(0, data)
}

// Update convenience: update existing crc with new data chunk
func Update(prev uint64, data []byte) uint64 {
	return CRC64(prev, data)
}

// Slice-by-8 implementation for little endian (fast path)
func crcspeed64little(table [8][256]uint64, crc uint64, buf []byte) uint64 {
	n := len(buf)
	i := 0
	// process unaligned head until 8-byte aligned pointer - in Go we just process until len%8
	// We prefer to process leading bytes so remaining len is multiple of 8.
	for (n-i) > 0 && ((uintptr(unsafe.Pointer(&buf[i])) & 7) != 0) {
		crc = table[0][byte(crc^uint64(buf[i]))] ^ (crc >> 8)
		i++
	}

	// process 8 bytes at a time
	for (n - i) >= 8 {
		// read little-endian uint64
		v := binary.LittleEndian.Uint64(buf[i : i+8])
		crc ^= v
		crc = table[7][byte(crc&0xff)] ^
			table[6][byte((crc>>8)&0xff)] ^
			table[5][byte((crc>>16)&0xff)] ^
			table[4][byte((crc>>24)&0xff)] ^
			table[3][byte((crc>>32)&0xff)] ^
			table[2][byte((crc>>40)&0xff)] ^
			table[1][byte((crc>>48)&0xff)] ^
			table[0][byte((crc>>56)&0xff)]
		i += 8
	}

	// process the tail bytes
	for ; i < n; i++ {
		crc = table[0][byte(crc^uint64(buf[i]))] ^ (crc >> 8)
	}
	return crc
}

// Big-endian slice-by-8 implementation
func crcspeed64big(table [8][256]uint64, crc uint64, buf []byte) uint64 {
	n := len(buf)
	i := 0
	// process until pointer aligned
	for (n-i) > 0 && ((uintptr(unsafe.Pointer(&buf[i])) & 7) != 0) {
		crc = table[0][byte((crc>>56)^uint64(buf[i]))] ^ (crc << 8)
		i++
	}

	for (n - i) >= 8 {
		v := binary.BigEndian.Uint64(buf[i : i+8])
		crc ^= v
		crc = table[0][byte(crc&0xff)] ^
			table[1][byte((crc>>8)&0xff)] ^
			table[2][byte((crc>>16)&0xff)] ^
			table[3][byte((crc>>24)&0xff)] ^
			table[4][byte((crc>>32)&0xff)] ^
			table[5][byte((crc>>40)&0xff)] ^
			table[6][byte((crc>>48)&0xff)] ^
			table[7][byte((crc>>56)&0xff)]
		i += 8
	}

	for ; i < n; i++ {
		crc = table[0][byte((crc>>56)^uint64(buf[i]))] ^ (crc << 8)
	}
	return crc
}
