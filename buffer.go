package blackcl

/*
#cgo CFLAGS: -I CL
#cgo !darwin LDFLAGS: -lOpenCL
#cgo darwin LDFLAGS: -framework OpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
*/
import "C"
import (
	"errors"
	"unsafe"
)

//buffer memory buffer on the device
type buffer struct {
	memobj C.cl_mem
	size   int
	device *Device
}

//newBuffer creates new buffer with specified size
func newBuffer(d *Device, size int) (C.cl_mem, error) {
	var ret C.cl_int
	clBuffer := C.clCreateBuffer(d.ctx, C.CL_MEM_READ_WRITE, C.size_t(size), nil, &ret)
	err := toErr(ret)
	if err != nil {
		return nil, err
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	return clBuffer, nil
}

//Release releases the buffer on the device
func (b *buffer) Release() error {
	return toErr(C.clReleaseMemObject(b.memobj))
}

func (b *buffer) copy(size int, ptr unsafe.Pointer) <-chan error {
	ch := make(chan error, 1)
	if b.size != size {
		ch <- errors.New("buffer size not equal to data len")
		return ch
	}
	var event C.cl_event
	err := toErr(C.clEnqueueWriteBuffer(
		b.device.queue,
		b.memobj,
		C.CL_FALSE,
		0,
		C.size_t(size),
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
		defer C.clReleaseEvent(event)
		ch <- toErr(C.clWaitForEvents(1, &event))
	}()
	return ch
}
