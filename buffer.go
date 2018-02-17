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

cl_image_desc* create_image_desc (
	cl_mem_object_type image_type,
	size_t image_width,
	size_t image_height,
	size_t image_depth,
	size_t image_array_size,
	size_t image_row_pitch,
	size_t image_slice_pitch,
	cl_uint num_mip_levels,
	cl_uint num_samples,
	cl_mem buffer
) {
	cl_image_desc *desc = malloc(sizeof(cl_image_desc));
	desc->image_type = image_type;
	desc->image_width = image_width;
	desc->image_height = image_height;
	desc->image_row_pitch = image_row_pitch;
	desc->image_slice_pitch = image_slice_pitch;
	desc->num_mip_levels = num_mip_levels;
	desc->num_samples = num_samples;
	desc->buffer = buffer;
	return desc;
}

*/
import "C"
import (
	"errors"
	"image"
	"unsafe"
)

//Buffer memory buffer on the device
type Buffer struct {
	memobj   C.cl_mem
	size     int
	device   *Device
	order    C.cl_channel_order
	rowPitch int
	bounds   image.Rectangle
}

//NewBuffer creates new buffer with specified size
func (d *Device) NewBuffer(size int) (*Buffer, error) {
	var ret C.cl_int
	clBuffer := C.clCreateBuffer(d.ctx, C.CL_MEM_READ_WRITE, C.size_t(size), nil, &ret)
	err := toErr(ret)
	if err != nil {
		return nil, err
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	return &Buffer{memobj: clBuffer, size: size, device: d}, nil
}

//NewBufferFromFloat32 allocates new buffer and copies specified data to it
func (d *Device) NewBufferFromFloat32(data []float32) (*Buffer, error) {
	buf, err := d.NewBuffer(len(data) * 4)
	if err != nil {
		return nil, err
	}
	err = <-buf.CopyFloat32(data)
	if err != nil {
		buf.Release()
		return nil, err
	}
	return buf, nil
}

//NewBufferFromImage creates an buffer and copies image data to it
func (d *Device) NewBufferFromImage(img image.Image) (*Buffer, error) {
	/*
		cl_mem_object_type image_type,
		size_t image_width,
		size_t image_height,
		size_t image_depth,
		size_t image_array_size,
		size_t image_row_pitch,
		size_t image_slice_pitch,
		cl_uint num_mip_levels,
		cl_uint num_samples,
		cl_mem buffer
	*/
	switch m := img.(type) {
	case *image.Gray:
		var format C.cl_image_format
		format.image_channel_order = C.CL_INTENSITY
		format.image_channel_data_type = C.CL_UNORM_INT8
		desc := C.create_image_desc(
			C.CL_MEM_OBJECT_IMAGE2D,
			C.size_t(m.Bounds().Dx()),
			C.size_t(m.Bounds().Dy()),
			0,
			0,
			C.size_t(m.Stride),
			0,
			0,
			0,
			nil)
		return d.createImage(format, desc, m.Bounds().Dx(), m.Bounds().Dy(), m.Stride, m.Pix)
	case *image.RGBA:
		var format C.cl_image_format
		format.image_channel_order = C.CL_RGBA
		format.image_channel_data_type = C.CL_UNORM_INT8
		desc := C.create_image_desc(
			C.CL_MEM_OBJECT_IMAGE2D,
			C.size_t(m.Bounds().Dx()),
			C.size_t(m.Bounds().Dy()),
			0,
			0,
			C.size_t(m.Stride),
			0,
			0,
			0,
			nil)
		return d.createImage(format, desc, m.Bounds().Dx(), m.Bounds().Dy(), m.Stride, m.Pix)
	}

	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	data := make([]byte, w*h*4)
	dataOffset := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.At(x+b.Min.X, y+b.Min.Y)
			r, g, b, a := c.RGBA()
			data[dataOffset] = uint8(r >> 8)
			data[dataOffset+1] = uint8(g >> 8)
			data[dataOffset+2] = uint8(b >> 8)
			data[dataOffset+3] = uint8(a >> 8)
			dataOffset += 4
		}
	}
	var format C.cl_image_format
	format.image_channel_order = C.CL_RGBA
	format.image_channel_data_type = C.CL_UNORM_INT8
	desc := C.create_image_desc(C.CL_MEM_OBJECT_IMAGE2D, C.size_t(w), C.size_t(h), 0, 1, 0, 0, 0, 0, nil)
	return d.createImage(format, desc, w, h, 0, data)
}

func (d *Device) createImage(format C.cl_image_format, desc *C.cl_image_desc, width, height, rowPitch int, data []byte) (*Buffer, error) {
	var dataPtr unsafe.Pointer
	if data != nil {
		dataPtr = unsafe.Pointer(&data[0])
	}
	var ret C.cl_int
	clBuffer := C.clCreateImage(d.ctx, C.CL_MEM_READ_WRITE|C.CL_MEM_COPY_HOST_PTR, &format, desc, dataPtr, &ret)
	err := toErr(ret)
	if err != nil {
		return nil, err
	}
	if clBuffer == nil {
		return nil, ErrUnknown
	}
	return &Buffer{
		memobj:   clBuffer,
		size:     len(data),
		bounds:   image.Rect(0, 0, width, height),
		order:    format.image_channel_order,
		rowPitch: rowPitch,
		device:   d,
	}, nil
}

//Release releases the buffer on the device
func (b *Buffer) Release() error {
	return toErr(C.clReleaseMemObject(b.memobj))
}

func (b *Buffer) copy(size int, ptr unsafe.Pointer) <-chan error {
	ch := make(chan error, 1)
	if b.size != size {
		ch <- errors.New("buffer size not equal to data len")
		return ch
	}
	var event C.cl_event
	err := toErr(C.clEnqueueWriteBuffer(
		b.device.queue,
		b.memobj,
		C.CL_FALSE,
		0,
		C.size_t(size),
		ptr,
		0,
		nil,
		&event,
	))
	if err != nil {
		ch <- err
		return ch
	}
	go func() {
		defer C.clReleaseEvent(event)
		ch <- toErr(C.clWaitForEvents(1, &event))
	}()
	return ch
}

//Copy copies the data from host data to device buffer
//it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Buffer) Copy(data []byte) <-chan error {
	return b.copy(len(data), unsafe.Pointer(&data[0]))
}

//CopyFloat32 copies the float32 data from host data to device buffer
//it's a non-blocking call, channel will return an error or nil if the data transfer is complete
func (b *Buffer) CopyFloat32(data []float32) <-chan error {
	return b.copy(len(data)*4, unsafe.Pointer(&data[0]))
}

//Data gets data from device, it's a blocking call
func (b *Buffer) Data() ([]byte, error) {
	data := make([]byte, b.size)
	err := toErr(C.clEnqueueReadBuffer(
		b.device.queue,
		b.memobj,
		C.CL_TRUE,
		0,
		C.size_t(b.size),
		unsafe.Pointer(&data[0]),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, err
	}
	return data, nil
}

//DataFloat32 gets float32 data from device, it's a blocking call
func (b *Buffer) DataFloat32() ([]float32, error) {
	data := make([]float32, b.size/4)
	err := toErr(C.clEnqueueReadBuffer(
		b.device.queue,
		b.memobj,
		C.CL_TRUE,
		0,
		C.size_t(b.size),
		unsafe.Pointer(&data[0]),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, err
	}
	return data, nil
}

//DataImage gets data from an image buffer and returns an image.Image
func (b *Buffer) DataImage() (image.Image, error) {
	data := make([]byte, b.size)
	cOrigin := make([]C.size_t, 3)
	cRegion := []C.size_t{C.size_t(b.bounds.Dx()), C.size_t(b.bounds.Dy()), 1}
	err := toErr(C.clEnqueueReadImage(
		b.device.queue,
		b.memobj,
		C.CL_TRUE,
		&cOrigin[0],
		&cRegion[0],
		C.size_t(b.rowPitch),
		0,
		unsafe.Pointer(&data[0]),
		0,
		nil,
		nil,
	))
	if err != nil {
		return nil, errors.New("cannot get buffer data: " + err.Error())
	}
	switch b.order {
	case C.CL_RGBA:
		img := image.NewRGBA(b.bounds)
		img.Pix = data
		return img, nil
	case C.CL_INTENSITY:
		img := image.NewGray(b.bounds)
		img.Pix = data
		return img, nil
	}
	return nil, errors.New("cannot get image data from the buffer, not an image buffer")
}

//Map applies an map kernel on all elements of the buffer
func (b *Buffer) Map(k Kernel) <-chan error {
	return k([]int{int(b.size)}, []int{1}, b)
}
