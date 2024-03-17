# highCL
This is fork of [blackcl](https://github.com/Dadido3/blackcl), I believe that is not about black magic, <br>
This package provide high level wrapper to OpenCL,
that means it is completly hiding call of OpenCL. This package use [pureCL](https://github.com/opencl-pure/pureCL) to call OpenCL WITHOUT CGO!!! And it is continue of [middleCL](https://github.com/opencl-pure/middleCL)<br>
Absence cgo means you do not need c compiler, it is powered by. [purego](https://github.com/ebitengine/purego) and inspired by [libopencl-stub](https://github.com/krrishnarraj/libopencl-stub) and [Zyko0's opencl](https://github.com/Zyko0/go-opencl).
Thank to all of them!
# goal
- easy to multiplatform (thank [purego](https://github.com/ebitengine/purego))
- easy find path (custumize path to openclLib shared library)
- easy to compile, we do not need cgo and not need knowing link to shared library
- try [purego](https://github.com/ebitengine/purego) and bring opencl on android without complicate link
- be high level and allow more than `[]float32` vectors
# not goal
- be faster as cgo version, [purego](https://github.com/ebitengine/purego) is using same mechanism as cgo 
# examples
## 1

```go
package main

import (
	"fmt"
	opencl "github.com/opencl-pure/highCL"
	pure "github.com/opencl-pure/pureCL"
	"log"
)

func main() {
	err := opencl.Init(pure.Version2_0) //init with version of OpenCL
	if err != nil {
		log.Fatal(err)
	}
	//Do not create platforms/devices/contexts/queues/...
	//Just get the GPU
	d, err := opencl.GetDefaultDevice()
	if err != nil {
		log.Fatal(err)
	}
	defer d.Release()

	data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} // []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	// has several kinds of device memory object: Bytes, Vector, Image
	//allocate buffer on the device (16 elems of float32)
	v, err := d.NewVector(data)
	if err != nil {
		log.Fatal(err)
	}
	defer v.Release()

	//an complicated kernel
	const kernelSource = `
__kernel void addOne(__global float* data) {
	const int i = get_global_id (0);
	data[i] += 1;
}
`

	//Add program source to device, get kernel
	_, err = d.AddProgram(fmt.Sprint(kernelSource))
	if err != nil {
		log.Fatal(err)
	}
	k, err := d.Kernel("addOne")
	if err != nil {
		log.Fatal(err)
	}
	//run kernel (global work size 16 and local work size 1)
	event, err := k.Global(16).Local(1).Run(nil, v)
	if err != nil {
		log.Fatal(err)
	}
	defer event.Release()

	//Get data from vector
	newData, err := v.Data()
	if err != nil {
		log.Fatal(err)
	}

	//prints out [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]
	fmt.Println(newData.Interface().([]float32))
}
```
## 2

```go
package main

import (
	"fmt"
	"github.com/opencl-pure/constantsCL"
	opencl "github.com/opencl-pure/highCL"
	pure "github.com/opencl-pure/pureCL"
	"image"
	"image/png"
	"log"
	"os"
)

func main() {
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

	//read image file
	imgFile, err := os.Open("test_data/opencl.png")
	if err != nil {
		log.Fatal(err)
	}
	i, _, err := image.Decode(imgFile)
	if err != nil {
		log.Fatal(err)
	}
	err = opencl.Init(pure.Version2_0) //init with version of OpenCL
	if err != nil {
		log.Fatal(err)
	}
	//Do not create platforms/devices/contexts/queues/...
	//Just get the GPU
	d, err := opencl.GetDefaultDevice()
	if err != nil {
		log.Fatal(err)
	}
	defer d.Release()
	//create image buffer
	img, err := d.NewImageFromImage2D(i)
	if err != nil {
		log.Fatal(err)
	}
	defer img.Release()
	//allocate an empty image for the result
	invertedImg, err := d.NewImage2D(constantsCL.CL_RGBA, img.Bounds())
	if err != nil {
		log.Fatal(err)
	}
	defer invertedImg.Release()
	_, err = d.AddProgram(fmt.Sprint(invertColorKernel))
	if err != nil {
		log.Fatal(err)
	}
	//invert colors of the image
	k, _ := d.Kernel("invert")
	// run kernel, and return an event
	event, err := k.Global(img.Bounds().Dx(), img.Bounds().Dy()).Local(1, 1).Run(nil, img, invertedImg)
	if err != nil {
		log.Fatal(err)
	}
	defer event.Release()
	//wait for the kernel to finish. Not really necessary here, this just serves as example
	event.Wait()
	//get the inverted image data and save it to a file
	inverted, err := invertedImg.Data()
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Create("inverted.png")
	if err != nil {
		log.Fatal(err)
	}
	png.Encode(f, inverted)
}
```
