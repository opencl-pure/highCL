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

//Kernel returns an kernel
//if retrieving the kernel didn't complete the function will panic
func (d *Device) Kernel(name string) *Kernel {
	cname := C.CString(name)
	var k C.cl_kernel
	var ret C.cl_int
	for _, p := range d.programs {
		k = C.clCreateKernel(p, cname, &ret)
		if ret == C.CL_INVALID_KERNEL_NAME {
			continue
		}
		if ret != C.CL_SUCCESS {
			panic(toErr(ret))
		}
		break
	}
	if ret == C.CL_INVALID_KERNEL_NAME {
		panic(fmt.Sprintf("kernel with name '%s' not found", name))
	}
	return newKernel(d, k)
}

//ErrUnsupportedArgumentType error
type ErrUnsupportedArgumentType struct {
	Index int
	Value interface{}
}

func (e ErrUnsupportedArgumentType) Error() string {
	return fmt.Sprintf("cl: unsupported argument type for index %d: %+v", e.Index, e.Value)
}

//Kernel represent an single kernel
type Kernel struct {
	d *Device
	k C.cl_kernel
}

//Global returns an kernel with global size set
func (k *Kernel) Global(globalWorkSizes ...int) KernelWithGlobal {
	return KernelWithGlobal{
		kernel:          k,
		globalWorkSizes: globalWorkSizes,
	}
}

//KernelWithGlobal is a kernel with the global size set
//to run the kernel it must also set the local size
type KernelWithGlobal struct {
	kernel          *Kernel
	globalWorkSizes []int
}

//Local sets the local work sizes and returns an KernelCall which takes kernel arguments and runs the kernel
func (kg KernelWithGlobal) Local(localWorkSizes ...int) KernelCall {
	return KernelCall{
		kernel:          kg.kernel,
		globalWorkSizes: kg.globalWorkSizes,
		localWorkSizes:  localWorkSizes,
	}
}

//KernelCall is a kernel with global and local work sizes set
//and it's ready to be run
type KernelCall struct {
	kernel          *Kernel
	globalWorkSizes []int
	localWorkSizes  []int
}

//Run calls the kernel on its device with specified global and local work sizes and arguments
//it's a non-blocking call, so it returns a channel that will send an error value when the kernel is done
//or nil if the call was successful
func (kc KernelCall) Run(args ...interface{}) <-chan error {
	ch := make(chan error, 1)
	err := kc.kernel.setArgs(args)
	if err != nil {
		ch <- err
		return ch
	}
	return kc.kernel.call(kc.globalWorkSizes, kc.localWorkSizes)
}

func releaseKernel(k *Kernel) {
	C.clReleaseKernel(k.k)
}

func newKernel(d *Device, k C.cl_kernel) *Kernel {
	kernel := &Kernel{d: d, k: k}
	runtime.SetFinalizer(kernel, releaseKernel)
	return kernel
}

func (k *Kernel) setArgs(args []interface{}) error {
	for i, arg := range args {
		if err := k.setArg(i, arg); err != nil {
			return err
		}
	}
	return nil
}

func (k *Kernel) setArg(index int, arg interface{}) error {
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

func (k *Kernel) setArgBuffer(index int, buffer *Buffer) error {
	mem := buffer.memobj
	return toErr(C.clSetKernelArg(k.k, C.cl_uint(index), C.size_t(unsafe.Sizeof(mem)), unsafe.Pointer(&mem)))
}

func (k *Kernel) setArgFloat32(index int, val float32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) setArgInt8(index int, val int8) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) setArgUint8(index int, val uint8) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) setArgInt32(index int, val int32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) setArgUint32(index int, val uint32) error {
	return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func (k *Kernel) setArgLocal(index int, size int) error {
	return k.setArgUnsafe(index, size, nil)
}

func (k *Kernel) setArgUnsafe(index, argSize int, arg unsafe.Pointer) error {
	return toErr(C.clSetKernelArg(k.k, C.cl_uint(index), C.size_t(argSize), arg))
}

func (k *Kernel) call(workSizes, lokalSizes []int) <-chan error {
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
