// program
package highCL

import (
	constants "github.com/opencl-pure/constantsCL"
	pure "github.com/opencl-pure/pureCL"
	"unsafe"
)

type Program struct {
	program pure.Program
}

// GetBinaries Return the program binaries associated with program.
func (p *Program) GetBinaries() ([][]byte, error) {
	var devices pure.Device
	err := pure.StatusToErr(pure.GetProgramInfo(p.program, pure.ProgramBuildInfo(constants.CL_PROGRAM_NUM_DEVICES), pure.Size(4), unsafe.Pointer(&devices), nil))
	if err != nil {
		return nil, err
	}
	deviceIDs := make([]pure.Device, devices)
	err = pure.StatusToErr(pure.GetProgramInfo(p.program, constants.CL_PROGRAM_DEVICES, pure.Size(len(deviceIDs)*4), unsafe.Pointer(&deviceIDs[0]), nil))
	if err != nil {
		return nil, err
	}
	binarySizes := make([]pure.Size, devices)
	err = pure.StatusToErr(pure.GetProgramInfo(p.program, constants.CL_PROGRAM_BINARY_SIZES, pure.Size(len(deviceIDs)*4), unsafe.Pointer(&binarySizes[0]), nil))
	if err != nil {
		return nil, err
	}

	binaries := make([][]byte, devices)
	cBinaries := make([]unsafe.Pointer, devices)
	for i, size := range binarySizes {
		cBinaries[i] = unsafe.Pointer(&make([]byte, size)[0])
	}
	err = pure.StatusToErr(pure.GetProgramInfo(p.program, constants.CL_PROGRAM_BINARIES, pure.Size(len(cBinaries)*4), unsafe.Pointer(&cBinaries[0]), nil))
	if err != nil {
		return nil, err
	}

	for i, size := range binarySizes {
		binaries[i] = (*(*[1 << 20]byte)(cBinaries[i]))[:size]
	}

	return binaries, nil
}
