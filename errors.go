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
)

var (
	//ErrUnknown Generally an unexpected result from an OpenCL function (e.g. CL_SUCCESS but null pointer)
	ErrUnknown = errors.New("cl: unknown error")
)

//ErrBlackCL converts the OpenCL error code to an go error
type ErrBlackCL C.cl_int

func (e ErrBlackCL) Error() string {
	if err, ok := errorMap[C.cl_int(e)]; ok {
		return err
	}
	return fmt.Sprintf("cl: error %d", int(e))
}

func toErr(code C.cl_int) error {
	if code == 0 {
		return nil
	}
	return ErrBlackCL(code)
}

//Common OpenCl errors
var (
	ErrDeviceNotFound                     = "cl: Device Not Found"
	ErrDeviceNotAvailable                 = "cl: Device Not Available"
	ErrCompilerNotAvailable               = "cl: Compiler Not Available"
	ErrMemObjectAllocationFailure         = "cl: Mem Object Allocation Failure"
	ErrOutOfResources                     = "cl: Out Of Resources"
	ErrOutOfHostMemory                    = "cl: Out Of Host Memory"
	ErrProfilingInfoNotAvailable          = "cl: Profiling Info Not Available"
	ErrMemCopyOverlap                     = "cl: Mem Copy Overlap"
	ErrImageFormatMismatch                = "cl: Image Format Mismatch"
	ErrImageFormatNotSupported            = "cl: Image Format Not Supported"
	ErrBuildProgramFailure                = "cl: Build Program Failure"
	ErrMapFailure                         = "cl: Map Failure"
	ErrMisalignedSubBufferOffset          = "cl: Misaligned Sub Buffer Offset"
	ErrExecStatusErrorForEventsInWaitList = "cl: Exec Status Error For Events In Wait List"
	ErrCompileProgramFailure              = "cl: Compile Program Failure"
	ErrLinkerNotAvailable                 = "cl: Linker Not Available"
	ErrLinkProgramFailure                 = "cl: Link Program Failure"
	ErrDevicePartitionFailed              = "cl: Device Partition Failed"
	ErrKernelArgInfoNotAvailable          = "cl: Kernel Arg Info Not Available"
	ErrInvalidValue                       = "cl: Invalid Value"
	ErrInvalidDeviceType                  = "cl: Invalid Device Type"
	ErrInvalidPlatform                    = "cl: Invalid Platform"
	ErrInvalidDevice                      = "cl: Invalid Device"
	ErrInvalidContext                     = "cl: Invalid Context"
	ErrInvalidQueueProperties             = "cl: Invalid Queue Properties"
	ErrInvalidCommandQueue                = "cl: Invalid Command Queue"
	ErrInvalidHostPtr                     = "cl: Invalid Host Ptr"
	ErrInvalidMemObject                   = "cl: Invalid Mem Object"
	ErrInvalidImageFormatDescriptor       = "cl: Invalid Image Format Descriptor"
	ErrInvalidImageSize                   = "cl: Invalid Image Size"
	ErrInvalidSampler                     = "cl: Invalid Sampler"
	ErrInvalidBinary                      = "cl: Invalid Binary"
	ErrInvalidBuildOptions                = "cl: Invalid Build Options"
	ErrInvalidProgram                     = "cl: Invalid Program"
	ErrInvalidProgramExecutable           = "cl: Invalid Program Executable"
	ErrInvalidKernelName                  = "cl: Invalid Kernel Name"
	ErrInvalidKernelDefinition            = "cl: Invalid Kernel Definition"
	ErrInvalidKernel                      = "cl: Invalid Kernel"
	ErrInvalidArgIndex                    = "cl: Invalid Arg Index"
	ErrInvalidArgValue                    = "cl: Invalid Arg Value"
	ErrInvalidArgSize                     = "cl: Invalid Arg Size"
	ErrInvalidKernelArgs                  = "cl: Invalid Kernel Args"
	ErrInvalidWorkDimension               = "cl: Invalid Work Dimension"
	ErrInvalidWorkGroupSize               = "cl: Invalid Work Group Size"
	ErrInvalidWorkItemSize                = "cl: Invalid Work Item Size"
	ErrInvalidGlobalOffset                = "cl: Invalid Global Offset"
	ErrInvalidEventWaitList               = "cl: Invalid Event Wait List"
	ErrInvalidEvent                       = "cl: Invalid Event"
	ErrInvalidOperation                   = "cl: Invalid Operation"
	ErrInvalidGlObject                    = "cl: Invalid Gl Object"
	ErrInvalidBufferSize                  = "cl: Invalid Buffer Size"
	ErrInvalidMipLevel                    = "cl: Invalid Mip Level"
	ErrInvalidGlobalWorkSize              = "cl: Invalid Global Work Size"
	ErrInvalidProperty                    = "cl: Invalid Property"
	ErrInvalidImageDescriptor             = "cl: Invalid Image Descriptor"
	ErrInvalidCompilerOptions             = "cl: Invalid Compiler Options"
	ErrInvalidLinkerOptions               = "cl: Invalid Linker Options"
	ErrInvalidDevicePartitionCount        = "cl: Invalid Device Partition Count"
)

var errorMap = map[C.cl_int]string{
	C.CL_SUCCESS:                                   "",
	C.CL_DEVICE_NOT_FOUND:                          ErrDeviceNotFound,
	C.CL_DEVICE_NOT_AVAILABLE:                      ErrDeviceNotAvailable,
	C.CL_COMPILER_NOT_AVAILABLE:                    ErrCompilerNotAvailable,
	C.CL_MEM_OBJECT_ALLOCATION_FAILURE:             ErrMemObjectAllocationFailure,
	C.CL_OUT_OF_RESOURCES:                          ErrOutOfResources,
	C.CL_OUT_OF_HOST_MEMORY:                        ErrOutOfHostMemory,
	C.CL_PROFILING_INFO_NOT_AVAILABLE:              ErrProfilingInfoNotAvailable,
	C.CL_MEM_COPY_OVERLAP:                          ErrMemCopyOverlap,
	C.CL_IMAGE_FORMAT_MISMATCH:                     ErrImageFormatMismatch,
	C.CL_IMAGE_FORMAT_NOT_SUPPORTED:                ErrImageFormatNotSupported,
	C.CL_BUILD_PROGRAM_FAILURE:                     ErrBuildProgramFailure,
	C.CL_MAP_FAILURE:                               ErrMapFailure,
	C.CL_MISALIGNED_SUB_BUFFER_OFFSET:              ErrMisalignedSubBufferOffset,
	C.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: ErrExecStatusErrorForEventsInWaitList,
	C.CL_INVALID_VALUE:                             ErrInvalidValue,
	C.CL_INVALID_DEVICE_TYPE:                       ErrInvalidDeviceType,
	C.CL_INVALID_PLATFORM:                          ErrInvalidPlatform,
	C.CL_INVALID_DEVICE:                            ErrInvalidDevice,
	C.CL_INVALID_CONTEXT:                           ErrInvalidContext,
	C.CL_INVALID_QUEUE_PROPERTIES:                  ErrInvalidQueueProperties,
	C.CL_INVALID_COMMAND_QUEUE:                     ErrInvalidCommandQueue,
	C.CL_INVALID_HOST_PTR:                          ErrInvalidHostPtr,
	C.CL_INVALID_MEM_OBJECT:                        ErrInvalidMemObject,
	C.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:           ErrInvalidImageFormatDescriptor,
	C.CL_INVALID_IMAGE_SIZE:                        ErrInvalidImageSize,
	C.CL_INVALID_SAMPLER:                           ErrInvalidSampler,
	C.CL_INVALID_BINARY:                            ErrInvalidBinary,
	C.CL_INVALID_BUILD_OPTIONS:                     ErrInvalidBuildOptions,
	C.CL_INVALID_PROGRAM:                           ErrInvalidProgram,
	C.CL_INVALID_PROGRAM_EXECUTABLE:                ErrInvalidProgramExecutable,
	C.CL_INVALID_KERNEL_NAME:                       ErrInvalidKernelName,
	C.CL_INVALID_KERNEL_DEFINITION:                 ErrInvalidKernelDefinition,
	C.CL_INVALID_KERNEL:                            ErrInvalidKernel,
	C.CL_INVALID_ARG_INDEX:                         ErrInvalidArgIndex,
	C.CL_INVALID_ARG_VALUE:                         ErrInvalidArgValue,
	C.CL_INVALID_ARG_SIZE:                          ErrInvalidArgSize,
	C.CL_INVALID_KERNEL_ARGS:                       ErrInvalidKernelArgs,
	C.CL_INVALID_WORK_DIMENSION:                    ErrInvalidWorkDimension,
	C.CL_INVALID_WORK_GROUP_SIZE:                   ErrInvalidWorkGroupSize,
	C.CL_INVALID_WORK_ITEM_SIZE:                    ErrInvalidWorkItemSize,
	C.CL_INVALID_GLOBAL_OFFSET:                     ErrInvalidGlobalOffset,
	C.CL_INVALID_EVENT_WAIT_LIST:                   ErrInvalidEventWaitList,
	C.CL_INVALID_EVENT:                             ErrInvalidEvent,
	C.CL_INVALID_OPERATION:                         ErrInvalidOperation,
	C.CL_INVALID_GL_OBJECT:                         ErrInvalidGlObject,
	C.CL_INVALID_BUFFER_SIZE:                       ErrInvalidBufferSize,
	C.CL_INVALID_MIP_LEVEL:                         ErrInvalidMipLevel,
	C.CL_INVALID_GLOBAL_WORK_SIZE:                  ErrInvalidGlobalWorkSize,
	C.CL_INVALID_PROPERTY:                          ErrInvalidProperty,
}
