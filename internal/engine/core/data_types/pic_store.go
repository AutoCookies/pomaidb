package data_types

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang/snappy"
)

// ErrKeyNotFound indicates the requested key or chain does not exist
var ErrChainNotFound = fmt.Errorf("inference chain not found")

// InferenceChain represents a single step in an LLM inference process
type InferenceChain struct {
	Prompt    string
	Response  []byte            // Compressed data if IsDelta=true, otherwise raw
	Metadata  map[string]string // Model, Temperature, TokenCount...
	Timestamp int64
	NextPred  string // Predicted next prompt (for preloading)
	IsDelta   bool   // True if Response is XOR-compressed against previous
}

// PICStore (Pomai Inference Cache) manages inference chains
type PICStore struct {
	mu     sync.RWMutex
	chains map[string][]InferenceChain // Key: ChainID (SessionID)
}

func NewPICStore() *PICStore {
	return &PICStore{
		chains: make(map[string][]InferenceChain),
	}
}

// Append adds a new inference step to a chain
// Performance: O(1) - No chain traversal, only compression against immediate tail
func (p *PICStore) Append(chainID, prompt string, response []byte, metadata map[string]string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	chains, ok := p.chains[chainID]
	if !ok {
		chains = make([]InferenceChain, 0, 10)
	}

	newItem := InferenceChain{
		Prompt:    prompt,
		Response:  response, // Default to raw
		Metadata:  metadata,
		Timestamp: time.Now().Unix(),
		IsDelta:   false,
	}

	if len(chains) > 0 {
		prev := chains[len(chains)-1]

		// 1. Predictive Preload Hint: Simple pattern matching on Prompts
		newItem.NextPred = predictNextPrompt(prev.Prompt, prompt)

		// 2. Chain Compression (XOR Delta + Snappy)
		// Optimization: We only attempt delta compression if the previous item is RAW.
		// If previous is already Delta, we store current as Raw to act as a new "checkpoint".
		// This avoids expensive recursive decompression during Append (keeping it O(1)).
		if !prev.IsDelta {
			compressedResponse := deltaCompress(prev.Response, response)

			// Only use delta if it actually saves space (snappy has overhead for small data)
			if len(compressedResponse) < len(response) {
				newItem.Response = compressedResponse
				newItem.IsDelta = true
			}
		}
	}

	chains = append(chains, newItem)
	p.chains[chainID] = chains
	return nil
}

// Get retrieves a specific inference step, handling decompression automatically
func (p *PICStore) Get(chainID string, idx int) (InferenceChain, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	chains, ok := p.chains[chainID]
	if !ok {
		return InferenceChain{}, ErrChainNotFound
	}

	if idx < 0 {
		idx = len(chains) + idx // Support negative index (-1 for latest)
	}

	if idx < 0 || idx >= len(chains) {
		return InferenceChain{}, fmt.Errorf("index out of bounds")
	}

	target := chains[idx]

	// Decompression Logic
	if target.IsDelta && idx > 0 {
		prev := chains[idx-1]

		// Since we enforce Raw->Delta structure in Append, prev is guaranteed (or highly likely) to be Raw.
		// If prev was somehow Delta (legacy data), reconstruction might fail or require recursive lookups.
		// Here we assume immediate dependency for O(1) Access.
		if !prev.IsDelta {
			decoded, err := deltaDecompress(prev.Response, target.Response)
			if err == nil {
				target.Response = decoded
				target.IsDelta = false // Return raw to user
			}
		}
	}

	return target, nil
}

// GetLatest returns the last item in the chain
func (p *PICStore) GetLatest(chainID string) (InferenceChain, error) {
	return p.Get(chainID, -1)
}

// --- Helpers ---

// XOR Delta + Snappy Compress
func deltaCompress(prev, curr []byte) []byte {
	minLen := len(prev)
	if len(curr) < minLen {
		minLen = len(curr)
	}

	// XOR the common parts
	diff := make([]byte, len(curr))
	for i := 0; i < minLen; i++ {
		diff[i] = prev[i] ^ curr[i]
	}
	// Copy the rest of curr as is
	copy(diff[minLen:], curr[minLen:])

	// Compress the XOR-ed result
	return snappy.Encode(nil, diff)
}

// Snappy Decompress + XOR Reconstruct
func deltaDecompress(prev, compressed []byte) ([]byte, error) {
	diff, err := snappy.Decode(nil, compressed)
	if err != nil {
		return nil, err
	}

	// Reconstruct: curr[i] = prev[i] ^ diff[i]
	curr := make([]byte, len(diff))
	minLen := len(prev)
	if len(diff) < minLen {
		minLen = len(diff)
	}

	for i := 0; i < minLen; i++ {
		curr[i] = prev[i] ^ diff[i]
	}
	copy(curr[minLen:], diff[minLen:])

	return curr, nil
}

// AI-Native Prediction Heuristics
func predictNextPrompt(prev, curr string) string {
	// 1. Numeric Pattern: "Step 1" -> "Step 2" -> Predict "Step 3"
	if strings.Contains(prev, "1") && strings.Contains(curr, "2") {
		return strings.Replace(curr, "2", "3", 1)
	}
	// 2. Code Pattern: "func A" -> "test A" -> Predict "bench A"
	if strings.HasPrefix(curr, "test ") {
		return strings.Replace(curr, "test ", "benchmark ", 1)
	}
	// 3. Conversational: "Hi" -> "How are you" -> Predict "What can you do"
	if len(curr) > len(prev) && len(curr) < 20 {
		return "continue"
	}
	return ""
}
