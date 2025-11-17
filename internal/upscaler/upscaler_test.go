package upscaler

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"runtime"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

// --- BENCHMARK SETUP ---

// TestMain is a special function that runs once for the entire test package.
// We use it to initialize and shut down the global ONNX Runtime environment.
func TestMain(m *testing.M) {
	// --- FIX: Initialize the ONNX Runtime for tests ---
	var libPath string
	if runtime.GOOS == "darwin" { // macOS
		// This is the correct Homebrew path for the ONNX Runtime shared library on Apple Silicon.
		libPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	} else { // Linux (for server deployment)
		libPath = "onnxruntime.so"
	}
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Environment for benchmarks: %v", err)
	}

	// Run all the tests and benchmarks in the package.
	exitCode := m.Run()

	// Shut down the ONNX Runtime environment.
	ort.DestroyEnvironment()
	os.Exit(exitCode)
}

// This helper function creates a sample image to be used in all benchmarks.
func setupBenchmarkImage() image.Image {
	// We use a 512x512 tile as that's our standard processing unit.
	width, height := 512, 512
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Fill with some dummy data
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x % 255), G: uint8(y % 255), B: 128, A: 255})
		}
	}
	return img
}

// --- PURE GO IMPLEMENTATION (for benchmarking) ---
func imageToFloatTensor_PureGo(img image.Image) (ort.Value, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	float32s := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			float32s[y*w+x] = float32(r>>8) / 255.0
			float32s[h*w+y*w+x] = float32(g>>8) / 255.0
			float32s[2*h*w+y*w+x] = float32(b>>8) / 255.0
		}
	}
	shape := ort.NewShape(1, 3, int64(h), int64(w))
	return ort.NewTensor(shape, float32s)
}

func floatTensorToImage_PureGo(tensor ort.Value) (image.Image, error) {
	t, ok := tensor.(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("output tensor is not of type float32")
	}
	shape := t.GetShape()
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 {
		return nil, errors.New("output tensor has unexpected shape")
	}
	h, w := int(shape[2]), int(shape[3])
	data := t.GetData()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := data[y*w+x] * 255.0
			g := data[h*w+y*w+x] * 255.0
			b := data[2*h*w+y*w+x] * 255.0
			img.Set(x, y, color.RGBA{
				R: clamp(r, 0, 255),
				G: clamp(g, 0, 255),
				B: clamp(b, 0, 255),
				A: 255,
			})
		}
	}
	return img, nil
}

// --- GOCV IMPLEMENTATION (for benchmarking) ---
func imageToFloatTensor_GoCV(img image.Image) (ort.Value, error) {
	mat, err := gocv.ImageToMatRGB(img)

	if err != nil {
		return nil, fmt.Errorf("failed to convert image to Mat: %w", err)
	}
	// Mat objects are C++ resources and must be closed to prevent memory leaks.
	defer mat.Close()

	// 2. Use BlobFromImage to perform all preprocessing steps at once.
	// This is significantly faster than manual conversion.
	// - 1.0/255.0: Normalizes pixels to the [0, 1] range.
	// - image.Point{X: mat.Cols(), Y: mat.Rows()}: Specifies the size (no change).
	// - gocv.NewScalar(0, 0, 0, 0): Specifies no mean subtraction.
	// - swapRB: false (because our Mat is already in RGB format).
	// - crop: false.
	blob := gocv.BlobFromImage(mat, 1.0/255.0, image.Point{X: mat.Cols(), Y: mat.Rows()}, gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blob.Close()

	// 3. Get the raw float data from the resulting blob.
	float32s, err := blob.DataPtrFloat32()
	if err != nil {
		return nil, fmt.Errorf("failed to get data pointer from blob: %w", err)
	}

	// 4. Create the final ONNX tensor.
	h, w := mat.Rows(), mat.Cols()
	shape := ort.NewShape(1, 3, int64(h), int64(w))
	// We need to copy the data because the blob's memory will be freed.
	tensorData := make([]float32, len(float32s))
	copy(tensorData, float32s)

	return ort.NewTensor(shape, tensorData)

}

func floatTensorToImage_GoCV(tensor ort.Value) (image.Image, error) {
	tensorFloat32, ok := tensor.(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("output tensor is not of type float32")
	}

	shape := tensorFloat32.GetShape()

	if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 {
		return nil, errors.New("output tensor has unexpected shape")
	}

	h, w := int(shape[2]), int(shape[3])
	float32s := tensorFloat32.GetData()
	singleChannelSize := h * w

	// 1. Create three separate single-channel Mats directly from the planar data.
	channels := make([]gocv.Mat, 3)
	for c := 0; c < 3; c++ {
		channelData := float32s[c*singleChannelSize : (c+1)*singleChannelSize]
		byteSlice := float32SliceToByteSlice(channelData)
		mat, err := gocv.NewMatFromBytes(h, w, gocv.MatTypeCV32F, byteSlice)
		if err != nil {
			return nil, fmt.Errorf("failed to create mat for channel %d: %w", c, err)
		}
		channels[c] = mat
	}

	// Ensure channel Mats are closed.
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()

	// 2. Merge the three planar channels into a single 3-channel (interleaved) Mat.
	mergedMat := gocv.NewMat()
	defer mergedMat.Close()
	gocv.Merge(channels, &mergedMat)

	// 3. De-normalize and convert data type in a single, efficient step.
	mergedMat.ConvertToWithParams(&mergedMat, gocv.MatTypeCV8UC3, 255.0, 0)

	// 4. Convert the final Mat back to a standard Go image.
	return mergedMat.ToImage()
}

// --- THE BENCHMARKS ---

func BenchmarkPureGoProcessing(b *testing.B) {
	img := setupBenchmarkImage()
	// Run the b.N times
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor, _ := imageToFloatTensor_PureGo(img)
		_, _ = floatTensorToImage_PureGo(tensor)
		tensor.Destroy()
	}
}

func BenchmarkGoCVProcessing(b *testing.B) {
	img := setupBenchmarkImage()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor, _ := imageToFloatTensor_GoCV(img)
		_, _ = floatTensorToImage_GoCV(tensor)
		tensor.Destroy()
	}
}
