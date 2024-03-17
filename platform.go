package highCL

import (
	constants "github.com/opencl-pure/constantsCL"
	pure "github.com/opencl-pure/pureCL"
	"strings"
)

type Platform struct {
	p pure.Platform
}

func getPlatforms() ([]*Platform, error) {
	numPlatforms := uint32(0)
	st := pure.GetPlatformIDs(0, nil, &numPlatforms)
	if st != constants.CL_SUCCESS {
		return nil, pure.StatusToErr(st)
	}
	platformIDs := make([]pure.Platform, numPlatforms)
	st = pure.GetPlatformIDs(numPlatforms, platformIDs, nil)
	if st != constants.CL_SUCCESS {
		return nil, pure.StatusToErr(st)
	}
	res := make([]*Platform, numPlatforms)
	for i := uint32(0); i < numPlatforms; i++ {
		res[i] = &Platform{p: platformIDs[i]}
	}
	return res, nil
}

func (p *Platform) getInfo(name pure.PlatformInfo) (string, error) {
	size := pure.Size(0)
	st := pure.GetPlatformInfo(p.p, name, pure.Size(0), nil, &size)
	if st != constants.CL_SUCCESS {
		return "", pure.StatusToErr(st)
	}

	info := make([]byte, size)
	st = pure.GetPlatformInfo(p.p, name, size, info, nil)
	if st != constants.CL_SUCCESS {
		return "", pure.StatusToErr(st)
	}
	return string(info[:size-1]), nil
}
func (p *Platform) GetProfile() (string, error) {
	return p.getInfo(constants.CL_PLATFORM_PROFILE)
}
func (p *Platform) GetVersion() (string, error) {
	return p.getInfo(constants.CL_PLATFORM_VERSION)
}
func (p *Platform) GetName() (string, error) {
	return p.getInfo(constants.CL_PLATFORM_NAME)
}
func (p *Platform) GetVendor() (string, error) {
	return p.getInfo(constants.CL_PLATFORM_VENDOR)
}
func (p *Platform) GetExtensions() ([]pure.Extension, error) {
	extensions, err := p.getInfo(constants.CL_PLATFORM_EXTENSIONS)
	if err != nil {
		return nil, err
	}
	return strings.Split(extensions, " "), nil
}
