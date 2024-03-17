package highCL

import (
	pure "github.com/opencl-pure/pureCL"
)

type Event struct {
	event pure.Event
}

// Wait on the host thread for commands identified by event objects to complete. Returns an error regarding the outcome of the associated task.
func (event *Event) Wait() error {
	list := []pure.Event{event.event}
	return pure.StatusToErr(pure.WaitForEvents(1, list))
}

// Decrements the event reference count.
func (event *Event) Release() error {
	return pure.StatusToErr(pure.ReleaseEvent(event.event))
}
