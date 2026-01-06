package hashtable

// HashObject là interface chung cho cả PackedHash và PPHT
type HashObject interface {
	Get(field string) ([]byte, bool)
	Set(field string, value []byte) (HashObject, bool) // Trả về (Object Mới, Có Upgrade hay không?)
	Delete(field string) (bool, int)                   // Trả về (Đã xóa?, Số lượng còn lại)
	Len() int
	GetAll() map[string][]byte
	IsPacked() bool
	SizeInBytes() int
}

// Cấu hình ngưỡng chuyển đổi
const (
	MaxPackedCount = 128  // Trên 128 phần tử -> Upgrade lên PPHT
	MaxPackedSize  = 4096 // Trên 4KB -> Upgrade lên PPHT
)
