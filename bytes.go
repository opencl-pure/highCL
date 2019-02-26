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
	"unsafe"
)

//Bytes is a memory buffer on the device that holds []byte
type Bytes struct {
	buf *buffer
}

//Size the size of the bytes buffer
func (b *Bytes) Size() int {
	return b.buf.size
}

//Release releases the buffer on the device
func (b *Bytes) Release() error {
	return b.buf.Release()
}

//NewBytes allocates new memory buffer with specified size on device
func (d *Device) NewBytes(size int) (*Bytes, error) {
	buf, err := newBuffer(d, size)
	if err != nil {
		return nil, err
	}
	return &Bytes{&buffer{memobj: buf, device: d, size: size}}, nil
}

//Copy copies the data from host data to device buffer
//it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Bytes) Copy(data []byte) <-chan error {
	return b.buf.copy(len(data), unsafe.Pointer(&data[0]))
}

//Data gets data from device, it's a blocking call
func (b *Bytes) Data() ([]byte, error) {
	data := make([]byte, b.buf.size)
	err := toErr(C.clEnqueueReadBuffer(
		b.buf.device.queue,
		b.buf.memobj,
		C.CL_TRUE,
		0,
		C.size_t(b.buf.size),
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
func (b *Bytes) Map(k *Kernel, returnEvent bool, waitEvents []*Event) (*Event, error) {
	return k.Global(b.buf.size).Local(1).Run(returnEvent, waitEvents, b)
}
