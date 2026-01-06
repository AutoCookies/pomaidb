package al

type Interface interface {
	Len() int
	Less(i, j int) bool
	Swap(i, j int)
}

func Sort(data Interface) {
	n := data.Len()
	quickSort(data, 0, n, maxDepth(n))
}

func maxDepth(n int) int {
	var depth int
	for i := n; i > 0; i >>= 1 {
		depth++
	}
	return depth * 2
}

func quickSort(data Interface, a, b, maxDepth int) {
	for b-a > 12 {
		if maxDepth == 0 {
			heapSort(data, a, b)
			return
		}
		maxDepth--
		mlo, mhi := doPivot(data, a, b)
		if mlo-a < b-mhi {
			quickSort(data, a, mlo, maxDepth)
			a = mhi
		} else {
			quickSort(data, mhi, b, maxDepth)
			b = mlo
		}
	}
	if b-a > 1 {
		for i := a + 1; i < b; i++ {
			for j := i; j > a && data.Less(j, j-1); j-- {
				data.Swap(j, j-1)
			}
		}
	}
}

func doPivot(data Interface, a, b int) (lo, hi int) {
	m := int(uint(a+b) >> 1)
	if b-a > 40 {
		s := (b - a) / 8
		medianOfThree(data, a, a+s, a+2*s)
		medianOfThree(data, m, m-s, m+s)
		medianOfThree(data, b-1, b-1-s, b-1-2*s)
	}
	medianOfThree(data, a, m, b-1)

	pivot := a
	a++
	c := b - 1

	for {
		for a < c && data.Less(a, pivot) {
			a++
		}
		for b > a && !data.Less(c, pivot) {
			c--
		}
		if a >= c {
			break
		}
		data.Swap(a, c)
		a++
		c--
	}

	data.Swap(pivot, c)
	return c, c + 1
}

func medianOfThree(data Interface, m1, m0, m2 int) {
	if data.Less(m1, m0) {
		data.Swap(m1, m0)
	}
	if data.Less(m2, m1) {
		data.Swap(m2, m1)
		if data.Less(m1, m0) {
			data.Swap(m1, m0)
		}
	}
}

func heapSort(data Interface, a, b int) {
	first := a
	lo := 0
	hi := b - a

	for i := (hi - 1) / 2; i >= 0; i-- {
		siftDown(data, i, hi, first)
	}

	for i := hi - 1; i >= 0; i-- {
		data.Swap(first, first+i)
		siftDown(data, lo, i, first)
	}
}

func siftDown(data Interface, root, len, offset int) {
	for {
		child := 2*root + 1
		if child >= len {
			break
		}
		if child+1 < len && data.Less(offset+child, offset+child+1) {
			child++
		}
		if !data.Less(offset+root, offset+child) {
			return
		}
		data.Swap(offset+root, offset+child)
		root = child
	}
}
