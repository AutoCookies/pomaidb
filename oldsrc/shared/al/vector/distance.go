package vector

import (
	"math"
)

type DistanceFunc func(a, b []float32) float32

func CosineSIMD(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 2.0
	}
	_ = a[len(a)-1]
	_ = b[len(b)-1]

	var dot, sumA, sumB float32

	i := 0
	for ; i <= len(a)-8; i += 8 {
		dot += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]

		sumA += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3] +
			a[i+4]*a[i+4] + a[i+5]*a[i+5] + a[i+6]*a[i+6] + a[i+7]*a[i+7]

		sumB += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3] +
			b[i+4]*b[i+4] + b[i+5]*b[i+5] + b[i+6]*b[i+6] + b[i+7]*b[i+7]
	}

	for ; i < len(a); i++ {
		dot += a[i] * b[i]
		sumA += a[i] * a[i]
		sumB += b[i] * b[i]
	}

	if sumA == 0 || sumB == 0 {
		return 1.0
	}

	return 1.0 - (dot / (float32(math.Sqrt(float64(sumA))) * float32(math.Sqrt(float64(sumB)))))
}

func EuclideanSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MaxFloat32)
	}

	dim := len(a)
	var sum float32

	i := 0
	for ; i+4 <= dim; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}

	for ; i < dim; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum)))
}

func DotProductSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	dim := len(a)
	var sum float32

	i := 0
	for ; i+4 <= dim; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}

	for ; i < dim; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

func ManhattanSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MaxFloat32)
	}

	dim := len(a)
	var sum float32

	i := 0
	for ; i+4 <= dim; i += 4 {
		d0 := a[i] - b[i]
		if d0 < 0 {
			d0 = -d0
		}
		d1 := a[i+1] - b[i+1]
		if d1 < 0 {
			d1 = -d1
		}
		d2 := a[i+2] - b[i+2]
		if d2 < 0 {
			d2 = -d2
		}
		d3 := a[i+3] - b[i+3]
		if d3 < 0 {
			d3 = -d3
		}
		sum += d0 + d1 + d2 + d3
	}

	for ; i < dim; i++ {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		sum += diff
	}

	return sum
}

func CosineDistance(a, b []float32) float32 {
	return CosineSIMD(a, b)
}

func EuclideanDistance(a, b []float32) float32 {
	return EuclideanSIMD(a, b)
}

func DotProduct(a, b []float32) float32 {
	return DotProductSIMD(a, b)
}

func ManhattanDistance(a, b []float32) float32 {
	return ManhattanSIMD(a, b)
}

func Normalize(vector []float32) []float32 {
	var norm float32
	for _, v := range vector {
		norm += v * v
	}

	if norm == 0 {
		return vector
	}

	norm = float32(math.Sqrt(float64(norm)))
	normalized := make([]float32, len(vector))
	for i, v := range vector {
		normalized[i] = v / norm
	}

	return normalized
}

func NormalizeInPlace(vector []float32) {
	var norm float32
	for _, v := range vector {
		norm += v * v
	}

	if norm == 0 {
		return
	}

	norm = float32(math.Sqrt(float64(norm)))
	for i := range vector {
		vector[i] /= norm
	}
}
