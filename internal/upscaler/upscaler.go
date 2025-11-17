package upscaler

import (
	"errors"
	"fmt"
	"image"
	"math"
	"runtime"
	"sync"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

// type tileTimings struct {
// 	id        int
// 	prepTime  time.Duration
// 	inferTime time.Duration
// 	postTime  time.Duration
// }

type Upscaler struct {
	session     *ort.DynamicAdvancedSession
	scale       int
	inputNames  []string
	outputNames []string
}

func NewUpscaler(modelPath string, scale int) (*Upscaler, error) {
	fmt.Printf("Loading ONNX model from: %s\n", modelPath)

	options, err := ort.NewSessionOptions()

	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()
	// coreml := map[string]string{
	// 	"MLComputeUnits":           "ALL",       // or "CPUAndGPU" if you want to avoid ANE
	// 	"ModelFormat":              "MLProgram", // better op coverage/perf on macOS 12+
	// 	"RequireStaticInputShapes": "0",
	// 	"EnableOnSubgraphs":        "0",
	// 	"ModelCacheDirectory":      "/tmp/ort-coreml-cache", // speeds up model load after first run
	// }
	err = options.AppendExecutionProviderCoreML(0)
	if err != nil {
		fmt.Println("Warning: could not enable CoreML Execution Provider. Falling back to CPU.", err)
	} else {
		fmt.Println("CoreML Execution Provider enabled for local GPU acceleration.")
	}

	inputNames := []string{"input"}
	outputNames := []string{"output"}

	// Create the session using the correct function signature.
	// We pass nil for the pre-allocated input/output tensors.
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create onnx session: %w", err)
	}

	fmt.Println("ONNX model loaded successfully.")
	return &Upscaler{
		session:     session,
		scale:       scale,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil

}

// Close releases the resources used by the ONNX session.
func (u *Upscaler) Close() {
	if u.session != nil {
		u.session.Destroy()
	}
}

// tileJob represents a single tile to be processed.
type tileJob struct {
	id        int
	x, y      int // Original coordinates
	tileInput image.Image
}

type tileResult struct {
	id           int
	x, y         int
	upscaledTile image.Image
	err          error
}

// ProcessImage upscales an image using a concurrent worker pool for tiled inference.
func (u *Upscaler) ProcessImage(img image.Image) (image.Image, error) {
	// Get Image size width*height
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// creating an empty image to fill after getting result from model
	outputImg := image.NewRGBA(image.Rect(0, 0, w*u.scale, h*u.scale))

	// Creating tiles to process the large images parallely as tiles and
	// adding padding of 10 so that there is stitching effect when tiles are combined back
	// also models are known to add artifacts at edges, so it reduces that effect also
	tileSize, tilePad := 512, 10
	tilesX := int(math.Ceil(float64(w) / float64(tileSize)))
	tilesY := int(math.Ceil(float64(h) / float64(tileSize)))
	// Creating 1 job for 1 tile
	numJobs := tilesX * tilesY
	fmt.Printf("Processing image in %d tiles using a concurrent worker pool...\n", numJobs)

	// this is a 3 part process
	// a prepWorker does calculations on cpu and provides the preparedtile to a batch worker
	// the batchworked creates a batch of tiles to process at once
	// once batch is created, all tensors in the batch are stacked so that the GPU resources can be used effectively
	// once the result comes in we unstack the tensors and create images for each tile and send in results channel
	// once all the tiles are stitched back together from results channel as 1 full image
	jobs := make(chan tileJob, numJobs)
	results := make(chan tileResult, numJobs)

	// --- BENCHMARKING: Channel to collect timing data ---
	// timings := make(chan tileTimings, numJobs)

	numWorkers := runtime.NumCPU()

	// very cpu intensive task and so more workers than cpu's dont make any impact

	// waitgroup to wait for prepworkers to finish prepping
	var wg sync.WaitGroup

	// calling prepworkers with jobs and prepared channels (cpu task)
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go u.worker(&wg, jobs, results)
	}

	// creating actual jobs, starting processing the subtiles and sending to prepworkers
	jobID := 0
	// for example 3*3 tiles
	// calculate start and end with padding to create subimage tiles
	// add jobid, start and end without pad and subimage into jobs channel
	// prepworker si waiting on jobs channel to prep for upscaling
	for y := 0; y < tilesY; y++ {
		for x := 0; x < tilesX; x++ {
			yStart, yEnd := y*tileSize, min((y+1)*tileSize, h)
			xStart, xEnd := x*tileSize, min((x+1)*tileSize, w)
			yStartPad, yEndPad := max(yStart-tilePad, 0), min(yEnd+tilePad, h)
			xStartPad, xEndPad := max(xStart-tilePad, 0), min(xEnd+tilePad, w)

			tileInputImg := img.(interface {
				SubImage(r image.Rectangle) image.Image
			}).SubImage(image.Rect(xStartPad, yStartPad, xEndPad, yEndPad))

			jobs <- tileJob{id: jobID, x: xStart, y: yStart, tileInput: tileInputImg}
			jobID++
		}
	}
	// closing jobs channel, if this is not called prepworkers will wait indefintely
	close(jobs)

	wg.Wait()
	close(results)
	// close(timings)

	// var totalPrep, totalInfer, totalPost time.Duration
	// for t := range timings {
	// 	totalPrep += t.prepTime
	// 	totalInfer += t.inferTime
	// 	totalPost += t.postTime
	// }
	// fmt.Println("\n--- BENCHMARK RESULTS ---")
	// fmt.Printf("Total Pre-processing time (all tiles): %v\n", totalPrep)
	// fmt.Printf("Total AI Inference time (all tiles):   %v\n", totalInfer)
	// fmt.Printf("Total Post-processing time (all tiles):%v\n", totalPost)
	// fmt.Println("-------------------------\n")

	// Loop through results and put the image together
	// will write a much clearer explanation later
	for result := range results {
		if result.err != nil {
			return nil, fmt.Errorf("error processing tile %d: %w", result.id, result.err)
		}

		xStart, yStart := result.x, result.y
		xEnd := min(xStart+tileSize, w)
		yEnd := min(yStart+tileSize, h)

		xStartPad := max(xStart-tilePad, 0)
		yStartPad := max(yStart-tilePad, 0)

		cropXStart := (xStart - xStartPad) * u.scale
		cropYStart := (yStart - yStartPad) * u.scale
		cropXEnd := cropXStart + (xEnd-xStart)*u.scale
		cropYEnd := cropYStart + (yEnd-yStart)*u.scale

		croppedOutputTile := result.upscaledTile.(interface {
			SubImage(r image.Rectangle) image.Image
		}).SubImage(image.Rect(cropXStart, cropYStart, cropXEnd, cropYEnd))

		pasteRect := image.Rect(xStart*u.scale, yStart*u.scale, xEnd*u.scale, yEnd*u.scale)
		drawTile(outputImg, pasteRect, croppedOutputTile)
	}

	fmt.Println("Image processing complete.")
	return outputImg, nil
}

func (u *Upscaler) worker(wg *sync.WaitGroup, jobs <-chan tileJob, results chan<- tileResult) {
	defer wg.Done()
	for job := range jobs {
		// prepStart := time.Now()
		inputTensor, err := imageToFloatTensor(job.tileInput)
		// prepTime := time.Since(prepStart)
		if err != nil {
			fmt.Printf("Error preparing tile %d: %v\n", job.id, err)
			continue
		}

		// --- UPDATED to match the correct function signature ---
		// The Run method expects a slice of inputs and a slice for outputs, and returns an error.
		inputs := []ort.Value{inputTensor}
		// We create a slice to hold the output. Passing nil for the element tells the
		// runtime to allocate the necessary memory for the output tensor.
		outputs := make([]ort.Value, 1)

		// --- BENCHMARKING: Time the AI inference step ---
		// inferStart := time.Now()
		err = u.session.Run(inputs, outputs)
		// inferTime := time.Since(inferStart)
		inputTensor.Destroy()
		if err != nil {
			results <- tileResult{id: job.id, err: fmt.Errorf("ONNX inference failed: %w", err)}
			continue
		}
		outputTensor := outputs[0]

		// --- BENCHMARKING: Time the post-processing step ---
		// postStart := time.Now()
		outputTileImg, err := floatTensorToImage(outputTensor)
		// postTime := time.Since(postStart)
		outputTensor.Destroy()
		if err != nil {
			results <- tileResult{id: job.id, err: err}
			continue
		}

		results <- tileResult{id: job.id, x: job.x, y: job.y, upscaledTile: outputTileImg}
		// --- BENCHMARKING: Send the timing results ---
		// timings <- tileTimings{id: job.id, prepTime: prepTime, inferTime: inferTime, postTime: postTime}
	}
}

func imageToFloatTensor(img image.Image) (ort.Value, error) {
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

// --- OPTIMIZED using GoCV for the reverse transformation ---
func floatTensorToImage(tensor ort.Value) (image.Image, error) {
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

// float32SliceToByteSlice performs an unsafe cast to avoid memory copies.
func float32SliceToByteSlice(floats []float32) []byte {
	if len(floats) == 0 {
		return nil
	}
	// Get a pointer to the first element of the float slice.
	ptr := unsafe.Pointer(&floats[0])
	// The length in bytes is the number of floats * 4 (the size of a float32).
	byteLen := len(floats) * 4
	// Create a new byte slice that points to the same underlying memory.
	return unsafe.Slice((*byte)(ptr), byteLen)
}

func clamp(v float32, lo, hi float32) uint8 {
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return uint8(v)
}

func drawTile(dst *image.RGBA, r image.Rectangle, src image.Image) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			dst.Set(x, y, src.At(x-r.Min.X+src.Bounds().Min.X, y-r.Min.Y+src.Bounds().Min.Y))
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
