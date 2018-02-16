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
	"fmt"
	"runtime"
	"unsafe"
)

//ErrUnsupportedArgumentType error
type ErrUnsupportedArgumentType struct {
	Index int
	Value interface{}
}

func (e ErrUnsupportedArgumentType) Error() string {
	return fmt.Sprintf("cl: unsupported argument type for index %d: %+v", e.Index, e.Value)
}

//Kernel is the specific kernel function
//calling the function sets the args of the kernel and calls the kernel on the device
//the Kernel remembers his args, so if an arg on another call is different from the arg before
//it will be changed and then called
//firts two arguments are workSizes and localSizes
type Kernel func([]int, []int, ...interface{}) <-chan error

type kernel struct {
	d    *Device
	k    C.cl_kernel
	args []interface{}
}

func releaseKernel(k *kernel) {
	C.clReleaseKernel(k.k)
}

func newKernel(d *Device, k C.cl_kernel) Kernel {
	kern := &kernel{d: d, k: k}
	runtime.SetFinalizer(kern, releaseKernel)
	return func(workSizes, lokalSizes []int, args ...interface{}) <-chan error {
		ch := make(chan error, 1)
		err := kern.setArgs(args)
		if err != nil {
			ch <- err
			return ch
		}
		return kern.call(workSizes, lokalSizes)
	}
}

func (k *kernel) setArgs(args []interface{}) error {
	for i, arg := range args {
		if k.args != nil && k.args[i] == arg {
			continue
		}
		err := k.setArg(i, arg)
		if err != nil {
			return err
		}
	}
	k.args = args
	return nil
}

func (k *kernel) setArg(index int, arg interface{}) error {
	switch val := arg.(type) {
	case uint8:
		return k.setArgUint8(index, val)
	case int8:
		return k.setArgInt8(index, val)
	case uint32:
		return k.setArgUint32(index, val)
	case int32:
		return k.setArgInt32(index, val)
	case float32:
		return k.setArgFloat32(index, val)
	case *Buffer:
		return k.setArgBuffer(index, val)
	//TODO case LocalBuffer:
	//	return k.setArgLocal(index, int(val))
	default:
		return ErrUnsupportedArgumentType{Index: index, Value: arg}
	}
}

func (k *kernel) setArgBuffer(index int, buffer *Buffer) error {
	mem := buffer.memobj
	return toErr(C.clSetKernelArg(k.k, C.cl_uint(index), C.size_t(unsafe.Sizeof(mem)), unsafe.Pointer(&mem)))
}

func (k *kernel) setArgFloat32(index int, val float32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *kernel) setArgInt8(index int, val int8) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *kernel) setArgUint8(index int, val uint8) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *kernel) setArgInt32(index int, val int32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *kernel) setArgUint32(index int, val uint32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *kernel) setArgLocal(index int, size int) error {
	return k.setArgUnsafe(index, size, nil)
}

func (k *kernel) setArgUnsafe(index, argSize int, arg unsafe.Pointer) error {
	return toErr(C.clSetKernelArg(k.k, C.cl_uint(index), C.size_t(argSize), arg))
}

func (k *kernel) call(workSizes, lokalSizes []int) <-chan error {
	ch := make(chan error, 1)
	workDim := len(workSizes)
	if workDim != len(lokalSizes) {
		ch <- errors.New("length of workSizes and localSizes differ")
		return ch
	}
	globalWorkOffsetPtr := make([]C.size_t, workDim)
	globalWorkSizePtr := make([]C.size_t, workDim)
	for i := 0; i < workDim; i++ {
		globalWorkSizePtr[i] = C.size_t(workSizes[i])
	}
	localWorkSizePtr := make([]C.size_t, workDim)
	for i := 0; i < workDim; i++ {
		localWorkSizePtr[i] = C.size_t(lokalSizes[i])
	}
	var event C.cl_event
	err := toErr(C.clEnqueueNDRangeKernel(
		k.d.queue,
		k.k,
		C.cl_uint(workDim),
		&globalWorkOffsetPtr[0],
		&globalWorkSizePtr[0],
		&localWorkSizePtr[0],
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
