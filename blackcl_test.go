package blackcl

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"testing"
)

func TestGetDevices(t *testing.T) {
	ds, err := GetDevices(DeviceTypeAll)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range ds {
		t.Log(d.Name())
		t.Log(d.Profile())
		t.Log(d.OpenCLCVersion())
		t.Log(d.DriverVersion())
		t.Log(d.Extensions())
		t.Log(d.Vendor())
		err = d.Release()
		if err != nil {
			t.Fatal(err)
		}
	}
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	if d == nil {
		t.Fatal("device is nil")
	}
	fmt.Println(d)
	err = d.Release()
	if err != nil {
		t.Fatal(err)
	}
}

func TestBuffer(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	buf, err := d.NewBuffer(16 * 4)
	if err != nil {
		t.Fatal(err)
	}
	data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = <-buf.CopyFloat32(data)
	if err != nil {
		t.Fatal(err)
	}
	retrievedData, err := buf.DataFloat32()
	if err != nil {
		t.Fatal(err)
	}
	if len(retrievedData) != len(data) {
		t.Fatal("data not same length")
	}
	for i := 0; i < 16; i++ {
		if data[i] != retrievedData[i] {
			t.Fatal("retrieved data not equal to sended data")
		}
	}
	err = buf.Release()
	if err != nil {
		t.Fatal(err)
	}
}

const testKernel = `
__kernel void testKernel(__global float* data) {
	const int i = get_global_id (0);
	data[i] += 1;
}
__kernel void testByteKernel(__global char* data) {
	const int i = get_global_id (0);
	data[i] += 1;
}
`

func TestBadProgram(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	defer recoverAddProgram(t)
	d.AddProgram("meh")
}

func recoverAddProgram(t *testing.T) {
	if err := recover(); err == nil {
		t.Fatal("not correct program compiled without error")
	}
}

func TestBadKernel(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	defer recoverKernel(t)
	d.AddProgram(testKernel)
	d.Kernel("meh")
}

func recoverKernel(t *testing.T) {
	if err := recover(); err == nil {
		t.Fatal("getting nonexisting kernel")
	}
}

func TestKernel(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	d.AddProgram(testKernel)
	k := d.Kernel("testKernel")
	buf, err := d.NewBuffer(16 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Release()
	data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = <-buf.CopyFloat32(data)
	if err != nil {
		t.Fatal(err)
	}
	err = <-k.Global(16).Local(1).Run(buf)
	if err != nil {
		t.Fatal(err)
	}
	receivedData, err := buf.DataFloat32()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if data[i]+1 != receivedData[i] {
			t.Error("receivedData not equal to data")
		}
	}
	err = <-buf.Map(k)
	if err != nil {
		t.Fatal(err)
	}
	receivedData, err = buf.DataFloat32()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if data[i]+2 != receivedData[i] {
			t.Error("receivedData not equal to data")
		}
	}
	err = d.Release()
	if err != nil {
		t.Fatal(err)
	}
}

func TestData(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	data := []byte(`abcdefgh`)
	buf, err := d.NewBuffer(len(data))
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Release()
	err = <-buf.Copy(data)
	if err != nil {
		t.Fatal(err)
	}
	d.AddProgram(testKernel)
	err = <-buf.Map(d.Kernel("testByteKernel"))
	if err != nil {
		t.Fatal(err)
	}
	receivedData, err := buf.Data()
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range data {
		if receivedData[i] != b+1 {
			t.Error("receivedData not equal to data")
		}
	}
}

const invertColorKernel = `
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void invert(__read_only image2d_t src, __write_only image2d_t dest) {
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 pixel = read_imagef(src, sampler, pos);
	pixel.x = 1 - pixel.x;
	pixel.y = 1 - pixel.y;
	pixel.z = 1 - pixel.z;
	write_imagef(dest, pos, pixel);
}`

func readImage(d *Device, path string) (image.Image, *Buffer, error) {
	imgFile, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		return nil, nil, err
	}
	buf, err := d.NewBufferFromImage(img)
	if err != nil {
		return nil, nil, err
	}
	return img, buf, nil
}

func writeImage(buf *Buffer, path string) error {
	receivedImg, err := buf.DataImage()
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, receivedImg, nil)
}

func TestImage(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	img, buf, err := readImage(d, "test_data/opencl.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Release()
	d.AddProgram(invertColorKernel)
	k := d.Kernel("invert")
	err = <-k.Global(img.Bounds().Dx(), img.Bounds().Dy()).Local(1, 1).Run(buf, buf)
	if err != nil {
		t.Fatal(err)
	}
	err = writeImage(buf, "/tmp/test.jpg")
	if err != nil {
		t.Fatal(err)
	}
	grayImg, grayBuf, err := readImage(d, "test_data/gopher.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer grayBuf.Release()
	err = <-k.Global(grayImg.Bounds().Dx(), grayImg.Bounds().Dy()).Local(1, 1).Run(grayBuf, grayBuf)
	if err != nil {
		t.Fatal(err)
	}
	err = writeImage(grayBuf, "/tmp/test_gray.jpg")
	if err != nil {
		t.Fatal(err)
	}
}

const gaussianBlurKernel = `
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 
__kernel void gaussian_blur(
        __read_only image2d_t image,
        __constant float * mask,
        __global float * blurredImage,
        __private int maskSize
    ) {
 
    const int2 pos = {get_global_id(0), get_global_id(1)};
 
    // Collect neighbor values and multiply with Gaussian
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            sum += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]
                *read_imagef(image, sampler, pos + (int2)(a,b)).x;
        }
    }
 
    blurredImage[pos.x+pos.y*get_global_size(0)] = sum;
}`

func TestGaussianBlur(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
}
