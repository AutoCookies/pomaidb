package ttl

import (
	"sync"
)

func parallelSumFloat32(data []float32, workers int) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	if workers <= 1 || n < 1024 {
		var s float64
		for _, v := range data {
			s += float64(v)
		}
		return s
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	res := make([]float64, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		i0 := w * chunk
		i1 := i0 + chunk
		if i1 > n {
			i1 = n
		}
		go func(idx, a, b int) {
			sum := 0.0
			for i := a; i < b; i++ {
				sum += float64(data[i])
			}
			res[idx] = sum
			wg.Done()
		}(w, i0, i1)
	}
	wg.Wait()
	total := 0.0
	for _, r := range res {
		total += r
	}
	return total
}

func parallelDotFloat32(a, b []float32, workers int) float64 {
	n := len(a)
	if n == 0 || len(b) != n {
		return 0
	}
	if workers <= 1 || n < 1024 {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += float64(a[i]) * float64(b[i])
		}
		return sum
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	res := make([]float64, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		i0 := w * chunk
		i1 := i0 + chunk
		if i1 > n {
			i1 = n
		}
		go func(idx, a0, b0 int) {
			sum := 0.0
			for i := a0; i < b0; i++ {
				sum += float64(a[i]) * float64(b[i])
			}
			res[idx] = sum
			wg.Done()
		}(w, i0, i1)
	}
	wg.Wait()
	total := 0.0
	for _, r := range res {
		total += r
	}
	return total
}
