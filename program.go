// program
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

type Program struct {
	program C.cl_program
}

// Return the program binaries associated with program.
func (p *Program) GetBinaries() ([][]byte, error) {

	var devices C.cl_uint

	err := toErr(C.clGetProgramInfo(p.program, C.CL_PROGRAM_NUM_DEVICES, C.size_t(C.sizeof_cl_uint), unsafe.Pointer(&devices), nil))
	if err != nil {
		return nil, err
	}
	deviceIDs := make([]C.cl_device_id, devices)
	err = toErr(C.clGetProgramInfo(p.program, C.CL_PROGRAM_DEVICES, C.size_t(len(deviceIDs)*C.sizeof_cl_device_id), unsafe.Pointer(&deviceIDs[0]), nil))
	if err != nil {
		return nil, err
	}
	binarySizes := make([]C.size_t, devices)
	err = toErr(C.clGetProgramInfo(p.program, C.CL_PROGRAM_BINARY_SIZES, C.size_t(len(deviceIDs)*C.sizeof_size_t), unsafe.Pointer(&binarySizes[0]), nil))
	if err != nil {
		return nil, err
	}

	binaries := make([][]byte, devices)
	cBinaries := make([]unsafe.Pointer, devices)
	for i, size := range binarySizes {
		cBinaries[i] = C.malloc(C.size_t(size))
		defer C.free(cBinaries[i])
	}
	err = toErr(C.clGetProgramInfo(p.program, C.CL_PROGRAM_BINARIES, C.size_t(len(cBinaries)*C.sizeof_size_t), unsafe.Pointer(&cBinaries[0]), nil))
	if err != nil {
		return nil, err
	}

	for i, size := range binarySizes {
		binaries[i] = C.GoBytes(unsafe.Pointer(cBinaries[i]), C.int(size))
	}

	return binaries, nil
}
