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

//Global returns an kernel with global offsets set
func (k *Kernel) GlobalOffset(globalWorkOffsets ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: globalWorkOffsets,
		globalWorkSizes:   []int{},
		localWorkSizes:    []int{},
	}
}

//Global returns an kernel with global offsets set
func (kc KernelCall) GlobalOffset(globalWorkOffsets ...int) KernelCall {
	kc.globalWorkOffsets = globalWorkOffsets
	return kc
}

//Global returns an KernelCall with global size set
func (k *Kernel) Global(globalWorkSizes ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: []int{},
		globalWorkSizes:   globalWorkSizes,
		localWorkSizes:    []int{},
	}
}

//Global returns an KernelCall with global size set
func (kc KernelCall) Global(globalWorkSizes ...int) KernelCall {
	kc.globalWorkSizes = globalWorkSizes
	return kc
}

//Local sets the local work sizes and returns an KernelCall which takes kernel arguments and runs the kernel
func (k *Kernel) Local(localWorkSizes ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: []int{},
		globalWorkSizes:   []int{},
		localWorkSizes:    localWorkSizes,
	}
}

//Local sets the local work sizes and returns an KernelCall which takes kernel arguments and runs the kernel
func (kc KernelCall) Local(localWorkSizes ...int) KernelCall {
	kc.localWorkSizes = localWorkSizes
	return kc
}

//KernelCall is a kernel with global and local work sizes set
//and it's ready to be run
type KernelCall struct {
	kernel            *Kernel
	globalWorkOffsets []int
	globalWorkSizes   []int
	localWorkSizes    []int
}

//Run calls the kernel on its device with specified global and local work sizes and arguments
//It's a non-blocking call, so it can return an event object that you can wait on.
//The caller is responsible to release the returned event when it's not used anymore.
func (kc KernelCall) Run(returnEvent bool, waitEvents []*Event, args ...interface{}) (event *Event, err error) {
	err = kc.kernel.setArgs(args)
	if err != nil {
		return
	}
	return kc.kernel.call(kc.globalWorkOffsets, kc.globalWorkSizes, kc.localWorkSizes, returnEvent, waitEvents)
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
	case *Bytes:
		return k.setArgBuffer(index, val.buf)
	case *Vector:
		return k.setArgBuffer(index, val.buf)
	case *Image:
		return k.setArgBuffer(index, val.buf)
	//TODO case LocalBuffer:
	//	return k.setArgLocal(index, int(val))
	default:
		return ErrUnsupportedArgumentType{Index: index, Value: arg}
	}
}

func (k *Kernel) setArgBuffer(index int, buf *buffer) error {
	mem := buf.memobj
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

func (k *Kernel) call(workOffsets, workSizes, lokalSizes []int, returnEvent bool, waitEvents []*Event) (event *Event, err error) {
	if len(workSizes) != len(lokalSizes) && len(lokalSizes) > 0 {
		err = errors.New("length of workSizes and localSizes differ")
		return
	}
	if len(workOffsets) > len(workSizes) {
		err = errors.New("workOffsets has a higher dimension than workSizes")
		return
	}
	globalWorkOffset := make([]C.size_t, len(workSizes))
	for i := 0; i < len(workOffsets); i++ {
		globalWorkOffset[i] = C.size_t(workOffsets[i])
	}
	globalWorkSize := make([]C.size_t, len(workSizes))
	for i := 0; i < len(workSizes); i++ {
		globalWorkSize[i] = C.size_t(workSizes[i])
	}
	localWorkSize := make([]C.size_t, len(lokalSizes))
	for i := 0; i < len(lokalSizes); i++ {
		localWorkSize[i] = C.size_t(lokalSizes[i])
	}
	cWaitEvents := make([]C.cl_event, len(waitEvents))
	for i := 0; i < len(waitEvents); i++ {
		cWaitEvents[i] = waitEvents[i].event
	}
	var waitEventsPtr *C.cl_event
	if len(cWaitEvents) > 0 {
		waitEventsPtr = &cWaitEvents[0]
	}
	var localWorkSizePtr unsafe.Pointer
	if len(lokalSizes) > 0 {
		localWorkSizePtr = unsafe.Pointer(&localWorkSize[0])
	}
	var eventPtr *C.cl_event
	if returnEvent {
		event = &Event{}
		eventPtr = &event.event
	}
	err = toErr(C.clEnqueueNDRangeKernel(
		k.d.queue,
		k.k,
		C.cl_uint(len(workSizes)),
		&globalWorkOffset[0],
		&globalWorkSize[0],
		(*C.size_t)(localWorkSizePtr),
		C.uint(len(waitEvents)),
		waitEventsPtr,
		eventPtr,
	))
	return
}
