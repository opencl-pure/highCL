package highCL

import (
	"errors"
	constants "github.com/opencl-pure/constantsCL"
	pure "github.com/opencl-pure/pureCL"
)

var (
	//ErrUnknown Generally an unexpected result from an OpenCL function (e.g. CL_SUCCESS but null pointer)
	ErrUnknown = errors.New("cl: unknown error")
)

// GetDefaultDevice ...
func GetDefaultDevice() (*Device, error) {
	id, err := GetDevices(pure.DeviceType(constants.CL_DEVICE_TYPE_DEFAULT))
	if err != nil {
		return nil, err
	}
	return id[0], nil
}

// GetDevices returns all devices of all platforms with specified type
func GetDevices(deviceType pure.DeviceType) ([]*Device, error) {
	platformIds, err := getPlatforms()
	if err != nil {
		return nil, err
	}
	var devices []*Device
	for _, p := range platformIds {
		var n uint32
		err = pure.StatusToErr(pure.GetDeviceIDs(p.p, deviceType, 0, nil, &n))
		if err != nil {
			return nil, err
		}
		deviceIds := make([]pure.Device, int(n))
		err = pure.StatusToErr(pure.GetDeviceIDs(p.p, deviceType, n, deviceIds, nil))
		if err != nil {
			return nil, err
		}
		for _, d := range deviceIds {
			device, err := newDevice([]pure.Device{d}, p)
			if err != nil {
				return nil, err
			}
			devices = append(devices, device)
		}
	}
	return devices, nil
}

func newDevice(id []pure.Device, p *Platform) (*Device, error) {
	d := &Device{id: id, platform: p}
	var ret pure.Status
	d.ctx = pure.CreateContext(nil, 1, id, nil, nil, &ret)
	err := pure.StatusToErr(ret)
	if err != nil {
		return nil, err
	}
	if d.ctx == pure.Context(0) {
		return nil, ErrUnknown
	}
	if pure.CreateCommandQueueWithProperties != nil {
		d.queue = pure.CreateCommandQueueWithProperties(d.ctx, d.id[0], 0, &ret)
	} else {
		d.queue = pure.CreateCommandQueue(d.ctx, d.id[0], 0, &ret)
	}
	if err = pure.StatusToErr(ret); err != nil {
		return nil, err
	}
	return d, nil
}

func SetHandle(handle uintptr) {
	pure.SetHandle(handle)
}

func Init(version pure.Version, paths ...string) error {
	return pure.Init(version, paths...)
}
