// event
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

type Event struct {
	event C.cl_event
}

// Waits on the host thread for commands identified by event objects to complete. Returns an error regarding the outcome of the associated task.
func (event *Event) Wait() error {
	return toErr(C.clWaitForEvents(1, &event.event))
}

// Decrements the event reference count.
func (event *Event) Release() {
	C.clReleaseEvent(event.event)
}
