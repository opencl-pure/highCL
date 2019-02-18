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

#ifdef CL_VERSION_2_0
int blackclOCLVersion = 2;
#else
int blackclOCLVersion = 1;
cl_command_queue clCreateCommandQueueWithProperties(
	cl_context context,
  	cl_device_id device,
  	void *properties,
  	cl_int *errcode_ret
) { return NULL; }
#endif
*/
import "C"

//GetDevices returns all devices of all platforms with specified type
func GetDevices(deviceType DeviceType) ([]*Device, error) {
	platformIds, err := getPlatforms()
	if err != nil {
		return nil, err
	}
	var devices []*Device
	for _, p := range platformIds {
		var n C.cl_uint
		err = toErr(C.clGetDeviceIDs(p, C.cl_device_type(deviceType), 0, nil, &n))
		if err != nil {
			return nil, err
		}
		deviceIds := make([]C.cl_device_id, int(n))
		err = toErr(C.clGetDeviceIDs(p, C.cl_device_type(deviceType), n, &deviceIds[0], nil))
		if err != nil {
			return nil, err
		}
		for _, d := range deviceIds {
			device, err := newDevice(d)
			if err != nil {
				return nil, err
			}
			devices = append(devices, device)
		}
	}
	return devices, nil
}

//GetDefaultDevice ...
func GetDefaultDevice() (*Device, error) {
	var id C.cl_device_id
	err := toErr(C.clGetDeviceIDs(nil, C.cl_device_type(DeviceTypeDefault), 1, &id, nil))
	if err != nil {
		return nil, err
	}
	return newDevice(id)
}

func getPlatforms() ([]C.cl_platform_id, error) {
	var n C.cl_uint
	err := toErr(C.clGetPlatformIDs(0, nil, &n))
	if err != nil {
		return nil, err
	}
	platformIds := make([]C.cl_platform_id, int(n))
	err = toErr(C.clGetPlatformIDs(n, &platformIds[0], nil))
	if err != nil {
		return nil, err
	}
	return platformIds, nil
}

func newDevice(id C.cl_device_id) (*Device, error) {
	d := &Device{id: id}
	var ret C.cl_int
	d.ctx = C.clCreateContext(nil, 1, &id, nil, nil, &ret)
	err := toErr(ret)
	if err != nil {
		return nil, err
	}
	if d.ctx == nil {
		return nil, ErrUnknown
	}
	if C.blackclOCLVersion == 2 {
		d.queue = C.clCreateCommandQueueWithProperties(d.ctx, d.id, nil, &ret)
	} else {
		d.queue = C.clCreateCommandQueue(d.ctx, d.id, 0, &ret)
	}
	err = toErr(ret)
	if err != nil {
		return nil, err
	}
	return d, nil
}
