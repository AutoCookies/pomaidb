package common

import "errors"

var (
	ErrEmptyKey            = errors.New("empty key")
	ErrInsufficientStorage = errors.New("insufficient storage")
	ErrValueNotInteger     = errors.New("value is not an integer")
	ErrKeyNotFound         = errors.New("key not found")
	ErrCorruptData         = errors.New("corrupted chunk data")
	ErrChainNotFound       = errors.New("inference chain not found")
	ErrDimensionMismatch   = errors.New("dimension mismatch")
)
