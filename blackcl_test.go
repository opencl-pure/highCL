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

func TestBytes(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	b, err := d.NewBytes(16)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = <-b.Copy(data)
	if err != nil {
		t.Fatal(err)
	}
	retrievedData, err := b.Data()
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
	err = b.Release()
	if err != nil {
		t.Fatal(err)
	}
}

func TestVector(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	v, err := d.NewVector(16)
	if err != nil {
		t.Fatal(err)
	}
	data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = <-v.Copy(data)
	if err != nil {
		t.Fatal(err)
	}
	retrievedData, err := v.Data()
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
	err = v.Release()
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
	v, err := d.NewVector(16)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Release()
	data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = <-v.Copy(data)
	if err != nil {
		t.Fatal(err)
	}
	err = <-k.Global(16).Local(1).Run(v)
	if err != nil {
		t.Fatal(err)
	}
	receivedData, err := v.Data()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if data[i]+1 != receivedData[i] {
			t.Error("receivedData not equal to data")
		}
	}
	err = <-v.Map(k)
	if err != nil {
		t.Fatal(err)
	}
	receivedData, err = v.Data()
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

func readImage(d *Device, path string) (*Image, error) {
	imgFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		return nil, err
	}
	i, err := d.NewImageFromImage(img)
	if err != nil {
		return nil, err
	}
	return i, nil
}

func writeImage(img *Image, path string) error {
	receivedImg, err := img.Data()
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
	img, err := readImage(d, "test_data/opencl.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer img.Release()
	d.AddProgram(invertColorKernel)
	k := d.Kernel("invert")
	invertedImg, err := d.NewImage(ImageTypeRGBA, img.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	err = <-k.Global(img.Bounds().Dx(), img.Bounds().Dy()).Local(1, 1).Run(img, invertedImg)
	if err != nil {
		t.Fatal(err)
	}
	err = writeImage(invertedImg, "/tmp/test.jpg")
	if err != nil {
		t.Fatal(err)
	}
	grayImg, err := readImage(d, "test_data/gopher.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer grayImg.Release()
	invertedGrayImg, err := d.NewImage(ImageTypeGray, grayImg.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	err = <-k.Global(grayImg.Bounds().Dx(), grayImg.Bounds().Dy()).Local(1, 1).Run(grayImg, invertedGrayImg)
	if err != nil {
		t.Fatal(err)
	}
	err = writeImage(invertedGrayImg, "/tmp/test_gray.jpg")
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
