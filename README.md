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
## 0
[examples dictionary](https://github.com/opencl-pure/highCL-examples):
|                                      |                                |                |                       |
| :----------------------------------: | :----------------------------: | :------------: | :-------------------: |
| ![Sierpinski Triangle](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/sierpinski_triangle_fractal.png) | ![Mandelbrot](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/mandelbrot_blue_red_black_fractal.png) | ![Julia](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/julia_fractal.png) | ![Mandelbrot Basic](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/mandelbrot_basic_fractal.png) |
| ![Mandelbrot Pseudo Random Colors](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/mandelbrot_pseudo_random_colors_fractal.png) | ![Sierpinski Triangle 2](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/sierpinski_triangle2_fractal.png) | ![Julia Set](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/julia_set_fractal.png) | ![Julia Basic](https://github.com/opencl-pure/highCL-examples/fill_image_fractals/outputs/julia_basic_fractal.png) |

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
	//init with version of OpenCL and variadic special paths (if you know)
	err := opencl.Init(pure.Version2_0/*, "some/special/path/opencl.dll", "some/special/path/opencl.so"*/)
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
