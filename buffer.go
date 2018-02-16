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

//Buffer memory buffer on the device
type Buffer struct {
	memobj C.cl_mem
	size   int
	device *Device
}

//NewBuffer creates new buffer with specified size
func (d *Device) NewBuffer(size int) (*Buffer, error) {
	var ret C.cl_int
	clBuffer := C.clCreateBuffer(d.ctx, C.CL_MEM_READ_WRITE, C.size_t(size), nil, &ret)
	err := toErr(ret)
	if err != nil {
		return nil, err
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	return &Buffer{memobj: clBuffer, size: size, device: d}, nil
}

//NewBufferFromFloat32 allocates new buffer and copies specified data to it
func (d *Device) NewBufferFromFloat32(data []float32) (*Buffer, error) {
	buf, err := d.NewBuffer(len(data) * 4)
	if err != nil {
		return nil, err
	}
	err = <-buf.CopyFloat32(data)
	if err != nil {
		buf.Release()
		return nil, err
	}
	return buf, nil
}

//Release releases the buffer on the device
func (b *Buffer) Release() error {
	return toErr(C.clReleaseMemObject(b.memobj))
}

func (b *Buffer) copy(size int, ptr unsafe.Pointer) <-chan error {
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

//Copy copies the data from host data to device buffer
//it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Buffer) Copy(data []byte) <-chan error {
	return b.copy(len(data), unsafe.Pointer(&data[0]))
}

//CopyFloat32 copies the float32 data from host data to device buffer
//it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Buffer) CopyFloat32(data []float32) <-chan error {
	return b.copy(len(data)*4, unsafe.Pointer(&data[0]))
}

//Data gets data from device, it's a blocking call
func (b *Buffer) Data() ([]byte, error) {
	data := make([]byte, b.size)
	err := toErr(C.clEnqueueReadBuffer(
		b.device.queue,
		b.memobj,
		C.CL_TRUE,
		0,
		C.size_t(b.size),
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

//DataFloat32 gets float32 data from device, it's a blocking call
func (b *Buffer) DataFloat32() ([]float32, error) {
	data := make([]float32, b.size/4)
	err := toErr(C.clEnqueueReadBuffer(
		b.device.queue,
		b.memobj,
		C.CL_TRUE,
		0,
		C.size_t(b.size),
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

//Map applies an map kernel on all elements of the buffer
func (b *Buffer) Map(k Kernel) <-chan error {
	return k([]int{int(b.size)}, []int{1}, b)
}
