package blackcl

import (
	"fmt"
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

func TestKernel(t *testing.T) {
	d, err := GetDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	if err = d.AddProgram("meh"); err == nil {
		t.Fatal("not correct program compiled without error")
	}
	if err = d.AddProgram(testKernel); err != nil {
		t.Fatalf("testKernel not compiled: %s", err)
	}
	_, err = d.Kernel("meh")
	if err == nil {
		t.Fatal("getting nonexisting kernel")
	}
	k, err := d.Kernel("testKernel")
	if err != nil {
		t.Fatal(err)
	}
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
	err = <-k([]int{16}, []int{1}, buf)
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
	k, _ := d.Kernel("testByteKernel")
	err = <-buf.Map(k)
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
	fmt.Println(string(receivedData))
}
