package data_types

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	al "github.com/AutoCookies/pomai-cache/shared/al/timestream"
	ds "github.com/AutoCookies/pomai-cache/shared/ds/timestream"
)

// TimeStreamStore quản lý nhiều Time Series Streams
type TimeStreamStore struct {
	streams sync.Map // map[string]*ds.Stream
	offsets sync.Map // map[string]int64 (group|stream -> offset)
}

func NewTimeStreamStore() *TimeStreamStore {
	return &TimeStreamStore{}
}

func (ts *TimeStreamStore) getOrCreateStream(name string) *ds.Stream {
	v, ok := ts.streams.Load(name)
	if ok {
		return v.(*ds.Stream)
	}

	s := ds.NewStream(24 * time.Hour) // Default retention
	actual, loaded := ts.streams.LoadOrStore(name, s)
	if loaded {
		return actual.(*ds.Stream)
	}
	return s
}

// Append: Wrapper gọi xuống DS
func (ts *TimeStreamStore) Append(streamName string, event *ds.Event) error {
	return ts.AppendBatch(streamName, []*ds.Event{event})
}

func (ts *TimeStreamStore) AppendBatch(streamName string, events []*ds.Event) error {
	s := ts.getOrCreateStream(streamName)
	s.Append(events)
	return nil
}

// Range: Truy vấn cơ bản
func (ts *TimeStreamStore) Range(streamName string, start, end int64, filterFunc func(*ds.Event) bool) ([]*ds.Event, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("stream not found")
	}
	s := v.(*ds.Stream)

	s.Mu.RLock()
	defer s.Mu.RUnlock()

	results := make([]*ds.Event, 0, 128)
	current := s.Head
	for current != nil {
		// MinMax Pruning
		if current.EndTime < start {
			current = current.Next
			continue
		}
		if current.StartTime > end {
			break
		}

		for i := 0; i < current.Count; i++ {
			ts := current.Timestamps[i]
			if ts >= start && ts <= end {
				// Lazy Hydration: Chỉ tạo object khi cần trả về
				// Có thể tối ưu thêm bằng cách check filterFunc trên raw column data trước khi hydrate
				e := current.GetEvent(i)
				if filterFunc == nil || filterFunc(e) {
					results = append(results, e)
				}
			}
		}
		current = current.Next
	}
	return results, nil
}

// Các hàm Compute: Gọi xuống sharedl

func (ts *TimeStreamStore) Window(streamName string, windowSize time.Duration, aggType string) (map[int64]float64, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("stream not found")
	}
	return al.Window(v.(*ds.Stream), windowSize, aggType)
}

func (ts *TimeStreamStore) DetectAnomaly(streamName string, threshold float64) ([]*ds.Event, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("stream not found")
	}
	return al.DetectAnomaly(v.(*ds.Stream), threshold)
}

func (ts *TimeStreamStore) Forecast(streamName string, horizon time.Duration) (float64, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return 0, fmt.Errorf("stream not found")
	}
	return al.Forecast(v.(*ds.Stream), horizon)
}

func (ts *TimeStreamStore) DetectPattern(streamName string, types []string, within time.Duration) ([][]*ds.Event, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("stream not found")
	}
	return al.DetectPattern(v.(*ds.Stream), types, within)
}

// Consumer Group Logic
func (ts *TimeStreamStore) ReadGroup(streamName, groupName string, count int) ([]*ds.Event, error) {
	v, ok := ts.streams.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("stream not found")
	}
	s := v.(*ds.Stream)

	lastOffset := ts.getOffset(streamName, groupName)

	// Gọi AL để đọc
	events, newOffset := al.ReadGroup(s, lastOffset, count)

	if len(events) > 0 {
		ts.setOffset(streamName, groupName, newOffset)
	}
	return events, nil
}

func (ts *TimeStreamStore) setOffset(stream, group string, timestamp int64) {
	key := group + "|" + stream
	ts.offsets.Store(key, timestamp)
}

func (ts *TimeStreamStore) getOffset(stream, group string) int64 {
	key := group + "|" + stream
	val, ok := ts.offsets.Load(key)
	if !ok {
		return 0
	}
	return val.(int64)
}

// Helpers
func ParseStreamAppend(payload []byte) ([]*ds.Event, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	var arr []*ds.Event
	if err := json.Unmarshal(payload, &arr); err == nil {
		out := make([]*ds.Event, 0, len(arr))
		for _, e := range arr {
			if e != nil {
				out = append(out, e)
			}
		}
		return out, nil
	}
	var single ds.Event
	if err := json.Unmarshal(payload, &single); err == nil {
		return []*ds.Event{&single}, nil
	}
	return nil, fmt.Errorf("invalid payload")
}
