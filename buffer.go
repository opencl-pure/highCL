package highCL

import (
	"errors"
	constants "github.com/opencl-pure/constantsCL"
	pure "github.com/opencl-pure/pureCL"
	"log"
	"unsafe"
)

// buffer memory buffer on the device
type buffer struct {
	memobj pure.Buffer
	size   pure.Size
	device *Device
}

// newBuffer creates new buffer with specified size
func newBuffer(d *Device, size int) (*buffer, error) {
	var ret pure.Status
	clBuffer := pure.CreateBuffer(d.ctx, constants.CL_MEM_READ_WRITE, pure.Size(size), nil, &ret)
	if err := pure.StatusToErr(ret); err != nil {
		return nil, err
	}
	if clBuffer == pure.Buffer(0) {
		return nil, ErrUnknown
	}
	return &buffer{
		memobj: clBuffer,
		size:   pure.Size(size),
		device: d,
	}, nil
}

// Release releases the buffer on the device
func (b *buffer) Release() error {
	return pure.StatusToErr(pure.ReleaseMemObject(b.memobj))
}

func (b *buffer) copy(size int, ptr unsafe.Pointer) <-chan error {
	ch := make(chan error, 1)
	if b.size != pure.Size(size) {
		ch <- errors.New("buffer size not equal to data len")
		return ch
	}
	var event pure.Event
	err := pure.StatusToErr(pure.EnqueueWriteBuffer(
		b.device.queue,
		b.memobj,
		false,
		0,
		pure.Size(size),
		ptr,
		0,
		nil,
		&event,
	))
	if err != nil {
		ch <- err
		return ch
	}
	go func() {
		list := []pure.Event{event}
		defer func() {
			if err2 := pure.StatusToErr(pure.ReleaseEvent(event)); err2 != nil {
				log.Println(err2)
			}
		}()
		ch <- pure.StatusToErr(pure.WaitForEvents(1, list))
	}()
	return ch
}
