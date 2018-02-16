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
	"fmt"
	"unsafe"
)

//DeviceType is an enum of device types
type DeviceType uint

//All values of DeviceType
const (
	DeviceTypeCPU         DeviceType = C.CL_DEVICE_TYPE_CPU
	DeviceTypeGPU         DeviceType = C.CL_DEVICE_TYPE_GPU
	DeviceTypeAccelerator DeviceType = C.CL_DEVICE_TYPE_ACCELERATOR
	DeviceTypeDefault     DeviceType = C.CL_DEVICE_TYPE_DEFAULT
	DeviceTypeAll         DeviceType = C.CL_DEVICE_TYPE_ALL
)

//Device the only needed entrence for the BlackCL
//represents the device on which memory can be allocated and kernels run
//it abstracts away all the complexity of contexts/platforms/queues
type Device struct {
	id       C.cl_device_id
	ctx      C.cl_context
	queue    C.cl_command_queue
	programs []C.cl_program
}

//Release releases the device
func (d *Device) Release() error {
	for _, p := range d.programs {
		err := toErr(C.clReleaseProgram(p))
		if err != nil {
			return err
		}
	}
	err := toErr(C.clReleaseCommandQueue(d.queue))
	if err != nil {
		return err
	}
	err = toErr(C.clReleaseContext(d.ctx))
	if err != nil {
		return err
	}
	return toErr(C.clReleaseDevice(d.id))
}

func (d *Device) getInfoString(param C.cl_device_info, panicOnError bool) (string, error) {
	var strC [1024]C.char
	var strN C.size_t
	err := toErr(C.clGetDeviceInfo(d.id, param, 1024, unsafe.Pointer(&strC), &strN))
	if err != nil {
		return "", err
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&strC)), C.int(strN)), nil
}

func (d *Device) String() string {
	return d.Name() + " " + d.Vendor()
}

//Name device info - name
func (d *Device) Name() string {
	str, _ := d.getInfoString(C.CL_DEVICE_NAME, true)
	return str
}

//Vendor device info - vendor
func (d *Device) Vendor() string {
	str, _ := d.getInfoString(C.CL_DEVICE_VENDOR, true)
	return str
}

//Extensions device info - extensions
func (d *Device) Extensions() string {
	str, _ := d.getInfoString(C.CL_DEVICE_EXTENSIONS, true)
	return str
}

//OpenCLCVersion device info - OpenCL C Version
func (d *Device) OpenCLCVersion() string {
	str, _ := d.getInfoString(C.CL_DEVICE_OPENCL_C_VERSION, true)
	return str
}

//Profile device info - profile
func (d *Device) Profile() string {
	str, _ := d.getInfoString(C.CL_DEVICE_PROFILE, true)
	return str
}

//Version device info - version
func (d *Device) Version() string {
	str, _ := d.getInfoString(C.CL_DEVICE_VERSION, true)
	return str
}

//DriverVersion device info - driver version
func (d *Device) DriverVersion() string {
	str, _ := d.getInfoString(C.CL_DRIVER_VERSION, true)
	return str
}

//AddProgram copiles program source
//if an error ocurres in building the program the AddProgram will panic
func (d *Device) AddProgram(source string) {
	var ret C.cl_int
	csource := C.CString(source)
	defer C.free(unsafe.Pointer(csource))
	p := C.clCreateProgramWithSource(d.ctx, 1, &csource, nil, &ret)
	err := toErr(ret)
	if err != nil {
		panic(err)
	}
	ret = C.clBuildProgram(p, 1, &d.id, nil, nil, nil)
	if ret != C.CL_SUCCESS {
		if ret == C.CL_BUILD_PROGRAM_FAILURE {
			var n C.size_t
			C.clGetProgramBuildInfo(p, d.id, C.CL_PROGRAM_BUILD_LOG, 0, nil, &n)
			log := make([]byte, int(n))
			C.clGetProgramBuildInfo(p, d.id, C.CL_PROGRAM_BUILD_LOG, n, unsafe.Pointer(&log[0]), nil)
			panic(string(log))
		}
		panic(toErr(ret))
	}
	d.programs = append(d.programs, p)
}

//Kernel returns an kernel function
//if retrieving the kernel didn't complete the function will panic
func (d *Device) Kernel(name string) Kernel {
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
