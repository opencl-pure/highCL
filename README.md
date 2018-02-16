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

//allocate buffer on the device (16 elems of float32)
buf, err := d.NewBuffer(16*4)
if err != nil {
	panic("could not allocate buffer")
}
defer buf.Release()

//copy data to the buffer (it's async)
data := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
err = <-buf.CopyFloat32(data)
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

//Add program source to device, get kernel and run kernel
d.AddProgram(kernelSource)
k := d.Kernel("addOne")
err = <-k(buf)
if err != nil {
	panic("could not run kernel")
}

//Get data from buffer
newData, err := buf.DataFloat32()
if err != nil {
	panic("could not get data from buffer")
}

//prints out [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]
fmt.Println(newData)

```
