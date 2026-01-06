package skiplist

import (
	"math/rand"
	"sync"
	"time"
)

const (
	maxLevel = 32
	p        = 0.25
)

type Element struct {
	Member string  `json:"member"`
	Score  float64 `json:"score"`
}

type Node struct {
	Member string
	Score  float64
	next   []*Node
	span   []int
}

type Skiplist struct {
	head   *Node
	level  int
	length int
	dict   map[string]float64
	mu     sync.RWMutex
}

func New() *Skiplist {
	return &Skiplist{
		head: &Node{
			next: make([]*Node, maxLevel),
			span: make([]int, maxLevel),
		},
		level: 1,
		dict:  make(map[string]float64),
	}
}

func randomLevel() int {
	level := 1
	for rand.Float64() < p && level < maxLevel {
		level++
	}
	return level
}

func (sl *Skiplist) Insert(member string, score float64) {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	if curScore, exists := sl.dict[member]; exists {
		if curScore == score {
			return
		}
		sl.deleteLocked(member, curScore)
	}

	sl.dict[member] = score
	update := make([]*Node, maxLevel)
	rank := make([]int, maxLevel)
	x := sl.head

	for i := sl.level - 1; i >= 0; i-- {
		if i == sl.level-1 {
			rank[i] = 0
		} else {
			rank[i] = rank[i+1]
		}
		for x.next[i] != nil && (x.next[i].Score < score || (x.next[i].Score == score && x.next[i].Member < member)) {
			rank[i] += x.span[i]
			x = x.next[i]
		}
		update[i] = x
	}

	lvl := randomLevel()
	if lvl > sl.level {
		for i := sl.level; i < lvl; i++ {
			rank[i] = 0
			update[i] = sl.head
			update[i].span[i] = sl.length
		}
		sl.level = lvl
	}

	x = &Node{
		Member: member,
		Score:  score,
		next:   make([]*Node, lvl),
		span:   make([]int, lvl),
	}

	for i := 0; i < lvl; i++ {
		x.next[i] = update[i].next[i]
		update[i].next[i] = x

		x.span[i] = update[i].span[i] - (rank[0] - rank[i])
		update[i].span[i] = (rank[0] - rank[i]) + 1
	}

	for i := lvl; i < sl.level; i++ {
		update[i].span[i]++
	}

	sl.length++
}

func (sl *Skiplist) Delete(member string) bool {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	score, exists := sl.dict[member]
	if !exists {
		return false
	}

	sl.deleteLocked(member, score)
	delete(sl.dict, member)
	return true
}

func (sl *Skiplist) deleteLocked(member string, score float64) {
	update := make([]*Node, maxLevel)
	x := sl.head
	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && (x.next[i].Score < score || (x.next[i].Score == score && x.next[i].Member < member)) {
			x = x.next[i]
		}
		update[i] = x
	}

	x = x.next[0]
	if x != nil && x.Score == score && x.Member == member {
		for i := 0; i < sl.level; i++ {
			if update[i].next[i] == x {
				update[i].span[i] += x.span[i] - 1
				update[i].next[i] = x.next[i]
			} else {
				update[i].span[i]--
			}
		}
		for sl.level > 1 && sl.head.next[sl.level-1] == nil {
			sl.level--
		}
		sl.length--
	}
}

func (sl *Skiplist) GetRank(member string) int {
	sl.mu.RLock()
	defer sl.mu.RUnlock()

	score, exists := sl.dict[member]
	if !exists {
		return -1
	}

	rank := 0
	x := sl.head
	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && (x.next[i].Score < score || (x.next[i].Score == score && x.next[i].Member <= member)) {
			rank += x.span[i]
			x = x.next[i]
		}
	}
	return rank - 1
}

func (sl *Skiplist) GetRange(start, stop int) []Element {
	sl.mu.RLock()
	defer sl.mu.RUnlock()

	if start < 0 {
		start = sl.length + start
	}
	if stop < 0 {
		stop = sl.length + stop
	}

	if start < 0 {
		start = 0
	}
	if start >= sl.length || start > stop {
		return nil
	}

	x := sl.head
	accumulated := 0
	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && accumulated+x.span[i] <= start {
			accumulated += x.span[i]
			x = x.next[i]
		}
	}

	x = x.next[0]

	limit := stop - start + 1
	result := make([]Element, 0, limit)

	for x != nil && limit > 0 {
		result = append(result, Element{Member: x.Member, Score: x.Score})
		x = x.next[0]
		limit--
	}
	return result
}

func (sl *Skiplist) GetScore(member string) (float64, bool) {
	sl.mu.RLock()
	defer sl.mu.RUnlock()
	val, ok := sl.dict[member]
	return val, ok
}

func (sl *Skiplist) Card() int {
	sl.mu.RLock()
	defer sl.mu.RUnlock()
	return sl.length
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
