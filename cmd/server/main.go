package main

import "C"

import (
	"fmt"
	"log"
	"runtime"

	"github.com/azadraja/image-upscale-api/internal/api"
	"github.com/azadraja/image-upscale-api/internal/upscaler"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	// --- FIX: Initialize the ONNX Runtime ---
	// This must be called once before any other onnxruntime functions are used.

	var libPath string
	if runtime.GOOS == "darwin" { // macOS
		// This is the correct Homebrew path for the ONNX Runtime shared library on Apple Silicon.
		libPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	} else { // Linux (for server deployment)
		libPath = "/usr/local/lib/libonnxruntime.so"
	}
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Environment: %v", err)
	}
	// We are now responsible for destroying the environment when the app closes.
	defer ort.DestroyEnvironment()
	// --- NEW: Initialize the Upscaler Service ---
	// We create one instance of the upscaler when the server starts.
	// This loads the AI model into memory once.
	upscalerServices := make(map[string]map[int]*upscaler.Upscaler)
	modelTypes := []string{"general", "anime", "restorative"}
	supportedScales := []int{2, 3, 4}
	for _, modelType := range modelTypes {
		upscalerServices[modelType] = make(map[int]*upscaler.Upscaler)
		for _, scale := range supportedScales {
			// This logic constructs the correct filename for each model.
			var modelFilename string
			switch modelType {
			case "anime":
				modelFilename = fmt.Sprintf("RealESRGAN_x4plus_anime_6B_x%d.onnx", scale)
			case "general":
				modelFilename = fmt.Sprintf("RealESRGAN_x4plus_x%d.onnx", scale)
			case "restorative":
				modelFilename = fmt.Sprintf("RealESRNet_x4plus_x%d.onnx", scale) // New model file
			}
			modelPath := fmt.Sprintf("models/%s", modelFilename)

			service, err := upscaler.NewUpscaler(modelPath, scale)
			if err != nil {
				log.Fatalf("Failed to initialize upscaler for type '%s' scale %d: %v", modelType, scale, err)
			}
			defer service.Close()
			upscalerServices[modelType][scale] = service
		}
	}
	router := api.SetupRouter(upscalerServices)
	fmt.Println("Server is running on http://localhost:8080")
	router.Run(":8080")
}
