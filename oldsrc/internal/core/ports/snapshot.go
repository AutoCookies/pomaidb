package ports

import "io"

type Serializable interface {
	SnapshotTo(w io.Writer) error
	RestoreFrom(r io.Reader) error
}

type Snapshotter interface {
	Snapshot(target Serializable) error
	Restore(target Serializable) error
}
