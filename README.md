# BlackCL

Black magic with OpenCL. These are highly opinionated OpenCL bindings for Go. It tries to make GPU computing easy, with some sugar abstraction, Go's concurency and channels.

```go
//Do not create platforms/devices/contexts/queues/...
//Just get the GPU
d, err := blackcl.GetDefaultDevice()
if err != nil {
	panic("no opencl device")
}
defer d.Release()

//BlackCL has several kinds of device memory object: Bytes, Vector, Image
//allocate buffer on the device (16 elems of float32)
v, err := d.NewVector(16)
if err != nil {
	panic("could not allocate buffer")
}
defer v.Release()

//copy data to the vector (it's async)
data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
err = <-v.Copy(data)
if err != nil {
	panic("could not copy data to buffer")
}

//an complicated kernel
const kernelSource = `
__kernel void addOne(__global float* data) {
	const int i = get_global_id (0);
	data[i] += 1;
}
`

//Add program source to device, get kernel
d.AddProgram(kernelSource)
k := d.Kernel("addOne")
//run kernel (global work size 16 and local work size 1)
err = <-k.Global(16).Local(1).Run(v)
if err != nil {
	panic("could not run kernel")
}

//Get data from vector
newData, err := v.Data()
if err != nil {
	panic("could not get data from buffer")
}

//prints out [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]
fmt.Println(newData)

```

`BlackCL` also supports the `image.Image` interface, for image manipulation:

```go
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
//create image buffer
img, err := d.NewImageFromImage(i)
if err != nil {
	log.Fatal(err)
}
defer img.Release()
//image result
invertedImg, err := d.NewImage(blackcl.ImageTypeRGBA, img.Bounds())
if err != nil {
	log.Fatal(err)
}
defer invertedImg.Release()
d.AddProgram(invertColorKernel)
//invert colors of the image
k := d.Kernel("invert")
err = <-k.Global(img.Bounds().Dx(), img.Bounds().Dy()).Local(1, 1).Run(img, invertedImg)
if err != nil {
	log.Fatal(err)
}
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
```
