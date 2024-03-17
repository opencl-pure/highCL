package highCL

import (
	pure "github.com/opencl-pure/pureCL"
	"unsafe"
)

// Bytes is a memory buffer on the device that holds []byte
type Bytes struct {
	buf *buffer
}

// Size the size of the bytes buffer
func (b *Bytes) Size() int {
	return int(b.buf.size)
}

// Release releases the buffer on the device
func (b *Bytes) Release() error {
	return b.buf.Release()
}

// NewBytes allocates new memory buffer with specified size on device
func (d *Device) NewBytes(size int) (*Bytes, error) {
	buf, err := newBuffer(d, size)
	if err != nil {
		return nil, err
	}
	return &Bytes{buf: buf}, nil
}

// Set copies the data from host data to device buffer
// it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Bytes) Set(data []byte) <-chan error {
	return b.buf.copy(len(data), unsafe.Pointer(&data[0]))
}

// Data gets data from device, it's a blocking call
func (b *Bytes) Data() ([]byte, error) {
	data := make([]byte, b.buf.size)
	err := pure.StatusToErr(pure.EnqueueReadBuffer(
		b.buf.device.queue,
		b.buf.memobj,
		true,
		0,
		b.buf.size,
		unsafe.Pointer(&data[0]),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, err
	}
	return data, nil
}

// Map applies an map kernel on all elements of the buffer
// It's a non-blocking call, so it can return an event object that you can wait on.
// The caller is responsible to release the returned event when it's not used anymore.
func (b *Bytes) Map(k *Kernel, waitEvents []*Event) (*Event, error) {
	return k.Global(int(b.buf.size)).Local(1).Run(waitEvents, b)
}
