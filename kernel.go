package highCL

import (
	"errors"
	"fmt"
	constants "github.com/opencl-pure/constantsCL"
	pure "github.com/opencl-pure/pureCL"
	"unsafe"
)

// Kernel returns an kernel
// if retrieving the kernel didn't complete the function will panic
func (d *Device) Kernel(name string) (*Kernel, error) {
	var k pure.Kernel
	var ret pure.Status
	for _, p := range d.programs {
		k = pure.CreateKernel(p, name, &ret)
		if ret == constants.CL_INVALID_KERNEL_NAME {
			continue
		}
		if ret != constants.CL_SUCCESS {
			return nil, pure.StatusToErr(ret)
		}
		break
	}
	if ret == constants.CL_INVALID_KERNEL_NAME {
		return nil, pure.StatusToErr(ret)
	}
	return newKernel(d, k), nil
}

// ErrUnsupportedArgumentType error
type ErrUnsupportedArgumentType struct {
	Index int
	Value interface{}
}

func (e ErrUnsupportedArgumentType) Error() string {
	return fmt.Sprintf("cl: unsupported argument type for index %d: %+v", e.Index, e.Value)
}

// Kernel represent an single kernel
type Kernel struct {
	d *Device
	k pure.Kernel
}

// Global returns an kernel with global offsets set
func (k *Kernel) GlobalOffset(globalWorkOffsets ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: globalWorkOffsets,
		globalWorkSizes:   []int{},
		localWorkSizes:    []int{},
	}
}

// Global returns an kernel with global offsets set
func (kc KernelCall) GlobalOffset(globalWorkOffsets ...int) KernelCall {
	kc.globalWorkOffsets = globalWorkOffsets
	return kc
}

// Global returns an KernelCall with global size set
func (k *Kernel) Global(globalWorkSizes ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: []int{},
		globalWorkSizes:   globalWorkSizes,
		localWorkSizes:    []int{},
	}
}

// Global returns an KernelCall with global size set
func (kc KernelCall) Global(globalWorkSizes ...int) KernelCall {
	kc.globalWorkSizes = globalWorkSizes
	return kc
}

// Local sets the local work sizes and returns an KernelCall which takes kernel arguments and runs the kernel
func (k *Kernel) Local(localWorkSizes ...int) KernelCall {
	return KernelCall{
		kernel:            k,
		globalWorkOffsets: []int{},
		globalWorkSizes:   []int{},
		localWorkSizes:    localWorkSizes,
	}
}

// Local sets the local work sizes and returns an KernelCall which takes kernel arguments and runs the kernel
func (kc KernelCall) Local(localWorkSizes ...int) KernelCall {
	kc.localWorkSizes = localWorkSizes
	return kc
}

// KernelCall is a kernel with global and local work sizes set
// and it's ready to be run
type KernelCall struct {
	kernel            *Kernel
	globalWorkOffsets []int
	globalWorkSizes   []int
	localWorkSizes    []int
}

// Run calls the kernel on its device with specified global and local work sizes and arguments
// It's a non-blocking call, so it can return an event object that you can wait on.
// The caller is responsible to release the returned event when it's not used anymore.
func (kc KernelCall) Run(waitEvents []*Event, args ...interface{}) (event *Event, err error) {
	err = kc.kernel.setArgs(args)
	if err != nil {
		return
	}
	return kc.kernel.call(kc.globalWorkOffsets, kc.globalWorkSizes, kc.localWorkSizes, waitEvents)
}

func (k *Kernel) ReleaseKernel() error {
	return pure.StatusToErr(pure.ReleaseKernel(k.k))
}

func (k *Kernel) Finish() error {
	return pure.StatusToErr(pure.FinishCommandQueue(k.d.queue))
}

func (k *Kernel) Flush() error {
	return pure.StatusToErr(pure.FlushCommandQueue(k.d.queue))
}

func newKernel(d *Device, k pure.Kernel) *Kernel {
	kernel := &Kernel{d: d, k: k}
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
	case float32, float64, uint8, int8, uint16, int16,
		uint32, int32, uint64, int64:
		return k.setArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
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
	return pure.StatusToErr(pure.SetKernelArg(k.k, uint32(index), pure.Size(unsafe.Sizeof(mem)), unsafe.Pointer(&mem)))
}

func (k *Kernel) setArgLocal(index int, size int) error {
	return k.setArgUnsafe(index, size, nil)
}

func (k *Kernel) setArgUnsafe(index, argSize int, arg unsafe.Pointer) error {
	return pure.StatusToErr(pure.SetKernelArg(k.k, uint32(index), pure.Size(argSize), arg))
}

func (k *Kernel) call(workOffsets, workSizes, lokalSizes []int, waitEvents []*Event) (event *Event, err error) {
	if len(workSizes) != len(lokalSizes) && len(lokalSizes) > 0 {
		err = errors.New("length of workSizes and localSizes differ")
		return
	}
	if len(workOffsets) > len(workSizes) {
		err = errors.New("workOffsets has a higher dimension than workSizes")
		return
	}
	globalWorkOffset := make([]pure.Size, len(workSizes))
	for i := 0; i < len(workOffsets); i++ {
		globalWorkOffset[i] = pure.Size(workOffsets[i])
	}
	globalWorkSize := make([]pure.Size, len(workSizes))
	for i := 0; i < len(workSizes); i++ {
		globalWorkSize[i] = pure.Size(workSizes[i])
	}
	localWorkSize := make([]pure.Size, len(lokalSizes))
	for i := 0; i < len(lokalSizes); i++ {
		localWorkSize[i] = pure.Size(lokalSizes[i])
	}
	cWaitEvents := make([]pure.Event, len(waitEvents))
	for i := 0; i < len(waitEvents); i++ {
		cWaitEvents[i] = waitEvents[i].event
	}
	if waitEvents == nil {
		cWaitEvents = nil
	}
	event = &Event{}
	err = pure.StatusToErr(pure.EnqueueNDRangeKernel(
		k.d.queue,
		k.k,
		uint(uint32(len(workSizes))),
		globalWorkOffset,
		globalWorkSize,
		localWorkSize,
		uint(uint32(len(waitEvents))),
		cWaitEvents,
		&event.event,
	))
	return
}
