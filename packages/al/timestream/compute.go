package timestream

import (
	"fmt"
	"math"
	"sort"
	"time"

	ds "github.com/AutoCookies/pomai-cache/packages/ds/timestream"
)

func Window(s *ds.Stream, windowSize time.Duration, aggType string) (map[int64]float64, error) {
	s.Mu.RLock()
	defer s.Mu.RUnlock()

	buckets := make(map[int64]struct {
		sum   float64
		count int64
		min   float64
		max   float64
	})

	windowNano := windowSize.Nanoseconds()
	if windowNano <= 0 {
		windowNano = 1
	}

	current := s.Head
	for current != nil {
		count := current.Count
		for i := 0; i < count; i++ {
			val := current.Values[i]
			ts := current.Timestamps[i]

			bucketKey := (ts / windowNano) * windowNano

			b := buckets[bucketKey]
			if b.count == 0 {
				b.min = val
				b.max = val
			} else {
				if val < b.min {
					b.min = val
				}
				if val > b.max {
					b.max = val
				}
			}
			b.sum += val
			b.count++
			buckets[bucketKey] = b
		}
		current = current.Next
	}

	result := make(map[int64]float64, len(buckets))
	for t, b := range buckets {
		var finalVal float64
		switch aggType {
		case "sum":
			finalVal = b.sum
		case "avg":
			if b.count > 0 {
				finalVal = b.sum / float64(b.count)
			}
		case "min":
			finalVal = b.min
		case "max":
			finalVal = b.max
		case "count":
			finalVal = float64(b.count)
		default:
			return nil, fmt.Errorf("unknown agg type: %s", aggType)
		}
		result[t] = finalVal
	}
	return result, nil
}

func DetectAnomaly(s *ds.Stream, threshold float64) ([]*ds.Event, error) {
	s.Mu.RLock()
	defer s.Mu.RUnlock()

	if s.TotalSize < 10 {
		return nil, nil
	}

	mean := s.Sum / float64(s.TotalSize)
	variance := (s.SqSum / float64(s.TotalSize)) - (mean * mean)
	if variance < 0 {
		variance = 0
	}
	stdDev := math.Sqrt(variance)
	if stdDev == 0 {
		stdDev = 1e-9
	}

	anomalies := make([]*ds.Event, 0)

	current := s.Head
	for current != nil {
		for i := 0; i < current.Count; i++ {
			val := current.Values[i]
			zScore := math.Abs((val - mean) / stdDev)
			if zScore > threshold {
				anomalies = append(anomalies, current.GetEvent(i))
			}
		}
		current = current.Next
	}
	return anomalies, nil
}

func Forecast(s *ds.Stream, horizon time.Duration) (float64, error) {
	s.Mu.RLock()
	defer s.Mu.RUnlock()

	n := float64(s.TotalSize)
	if n < 2 {
		return 0, nil
	}

	var sumX, sumY, sumXY, sumXX float64
	baseTime := float64(s.Head.StartTime)
	var lastTime float64

	current := s.Head
	for current != nil {
		for i := 0; i < current.Count; i++ {
			x := float64(current.Timestamps[i]) - baseTime
			y := current.Values[i]

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
			lastTime = x
		}
		current = current.Next
	}

	denom := n*sumXX - sumX*sumX
	if math.Abs(denom) < 1e-9 {
		return 0, nil
	}

	slope := (n*sumXY - sumX*sumY) / denom
	intercept := (sumY - slope*sumX) / n

	futureTime := lastTime + float64(horizon.Nanoseconds())
	return slope*futureTime + intercept, nil
}

func ReadGroup(s *ds.Stream, lastReadTimestamp int64, count int) ([]*ds.Event, int64) {
	s.Mu.RLock()
	defer s.Mu.RUnlock()

	results := make([]*ds.Event, 0, count)
	current := s.Head
	newOffset := lastReadTimestamp

	for current != nil {
		// Skip blocks completely older than lastRead
		if current.EndTime <= lastReadTimestamp {
			current = current.Next
			continue
		}

		// Binary Search inside block for starting index
		idx := sort.Search(current.Count, func(i int) bool {
			return current.Timestamps[i] > lastReadTimestamp
		})

		for i := idx; i < current.Count; i++ {
			if len(results) >= count {
				return results, newOffset
			}
			ev := current.GetEvent(i)
			results = append(results, ev)
			newOffset = ev.Timestamp
		}
		current = current.Next
	}
	return results, newOffset
}

func DetectPattern(s *ds.Stream, types []string, within time.Duration) ([][]*ds.Event, error) {
	if len(types) == 0 {
		return nil, nil
	}

	// Pre-lookup IDs to avoid string comparison in tight loop
	targetIDs := make([]uint16, len(types))
	for i, t := range types {
		targetIDs[i] = ds.GlobalTypeRegistry.GetID(t)
		if targetIDs[i] == 0 {
			return nil, nil
		}
	}

	s.Mu.RLock()
	defer s.Mu.RUnlock()

	matches := make([][]*ds.Event, 0)
	windowNano := within.Nanoseconds()
	if windowNano <= 0 {
		windowNano = math.MaxInt64
	}

	current := s.Head
	for current != nil {
		for i := 0; i < current.Count; i++ {
			// Fast Integer Comparison
			if current.TypeIDs[i] != targetIDs[0] {
				continue
			}

			firstTs := current.Timestamps[i]
			match := make([]*ds.Event, 0, len(types))
			match = append(match, current.GetEvent(i))

			typeIdx := 1
			cur2 := current
			j := i + 1
		FOUND_LOOP:
			for cur2 != nil {
				for ; j < cur2.Count; j++ {
					if cur2.TypeIDs[j] == targetIDs[typeIdx] {
						ev := cur2.GetEvent(j)
						match = append(match, ev)
						typeIdx++
						if typeIdx == len(types) {
							if ev.Timestamp-firstTs <= windowNano {
								matches = append(matches, match)
							}
							break FOUND_LOOP
						}
					}
				}
				cur2 = cur2.Next
				j = 0
			}
		}
		current = current.Next
	}
	return matches, nil
}
