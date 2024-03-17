package highCL

import (
	"errors"
	pure "github.com/opencl-pure/pureCL"
	"reflect"
	"unsafe"
)

// Vector is a memory buffer on device that holds []float32
type Vector struct {
	buf        *buffer
	iSize, len int
	typ        reflect.Type
}

// Length the length of the vector
func (v *Vector) Length() int {
	return v.len
}

// Release releases the buffer on the device
func (v *Vector) Release() error {
	return v.buf.Release()
}

// NewVector want slice or array to create opencl vector in gpu
// I highly recommend primitive types such as int, uint, float32, uint8, ...,
// but you are free to experiment with GO structs, but you must keep in mind,
// that is there no guarantee how OpenCL will pass them
func (d *Device) NewVector(data interface{}) (*Vector, error) {
	dataType := reflect.TypeOf(data)
	if dataType.Kind() != reflect.Slice && dataType.Kind() != reflect.Array {
		return nil, errors.New("data must be slice")
	}
	slice := reflect.ValueOf(data)
	sliceLen := slice.Len()
	if sliceLen == 0 {
		return nil, errors.New("slice must have at least 1 item")
	}
	iSize := int(dataType.Elem().Size())
	size := sliceLen * iSize
	buf, err := newBuffer(d, size)
	if err != nil {
		return nil, err
	}
	v := &Vector{buf: buf, iSize: iSize, len: sliceLen, typ: dataType}
	err = <-v.buf.copy(size, unsafe.Pointer(slice.Pointer()))
	if err != nil {
		_ = v.Release()
		return nil, err
	}
	return v, nil
}

// Reset want equal data as NewVector was given (slice or array), it must have equal length as vector
// it is usefully for recall kernel with others data without locate new vector
// it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (v *Vector) Reset(data interface{}) <-chan error {
	dataType := reflect.TypeOf(data)
	if dataType != v.typ {
		ch := make(chan error, 1)
		ch <- errors.New("data must be slice equal type as been created")
		return ch
	}
	slice := reflect.ValueOf(data)
	l := slice.Len()
	if v.Length() != l {
		ch := make(chan error, 1)
		ch <- errors.New("vector length not equal to data length")
		return ch
	}
	return v.buf.copy(l*v.iSize, unsafe.Pointer(slice.Pointer()))
}

// Data gets data *reflect.Value in from device, it's a blocking call
// use v, err := Data(); elen := any(retrievedData.Index(i).Float()) ...
func (v *Vector) Data() (*reflect.Value, error) {
	data := reflect.MakeSlice(v.typ, v.len, v.len)
	err := pure.StatusToErr(pure.EnqueueReadBuffer(
		v.buf.device.queue,
		v.buf.memobj,
		true,
		0,
		v.buf.size,
		unsafe.Pointer(data.Pointer()),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, err
	}
	return &data, nil
}

// DataArray gets data *reflect.Value in from device, it's a blocking call
// use v, err := DataArray(); array := *(v.Interface().(*[16]float32))
func (v *Vector) DataArray() (*reflect.Value, error) {
	data := reflect.New(reflect.ArrayOf(v.len, v.typ.Elem()))
	err := pure.StatusToErr(pure.EnqueueReadBuffer(
		v.buf.device.queue,
		v.buf.memobj,
		true,
		0,
		v.buf.size,
		unsafe.Pointer(data.Pointer()),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, err
	}
	return &data, nil
}

// Map applies an map kernel on all elements of the vector
// It's a non-blocking call, so it can return an event object that you can wait on.
// The caller is responsible to release the returned event when it's not used anymore.
func (v *Vector) Map(k *Kernel, waitEvents []*Event) (*Event, error) {
	return k.Global(v.Length()).Local(1).Run(waitEvents, v)
}
