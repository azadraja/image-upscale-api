package api

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg" // Import to register JPEG decoder
	"image/png"
	_ "image/png"
	"io"
	"net/http"
	"strconv"

	"github.com/azadraja/image-upscale-api/internal/upscaler"
	_ "github.com/jdeng/goheif"

	"github.com/gin-gonic/gin"
)

// handleUpscale is the function that processes the image upload.
// It now lives in its own file to keep the logic separated.
func HandleUpscale(c *gin.Context) {
	// Get the nested map of services from the context.
	up, exists := c.Get("upscalers")
	if !exists {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Upscaler services not available"})
		return
	}
	upscalerServices := up.(map[string]map[int]*upscaler.Upscaler)
	// 1. Get and validate model_type and scale from the user's request.
	modelType := c.DefaultPostForm("model_type", "general")
	if modelType != "general" && modelType != "anime" && modelType != "restorative" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model_type must be 'general', 'anime', or 'restorative'"})
		return
	}

	scaleStr := c.DefaultPostForm("scale", "4")
	scale, err := strconv.Atoi(scaleStr)
	if err != nil || (scale < 2 || scale > 4) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "scale must be an integer between 2 and 4"})
		return
	}

	// 2. Look up the specific model from the nested map.
	scaleMap, ok := upscalerServices[modelType]
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "model type not found on server"})
		return
	}
	upscalerService, ok := scaleMap[scale]
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("unsupported scale for model type '%s'", modelType)})
		return
	}

	// 2. Get the uploaded image file from the form data.
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "image file is required"})
		return
	}

	// 3. Open the uploaded file.
	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to open uploaded file"})
		return
	}
	defer src.Close()

	// 4. Read the file data into memory.
	imageData, err := io.ReadAll(src)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read image data"})
		return
	}

	// 5. Decode the image data to get its format and dimensions.
	// Because we added the goheif import, this function can now handle .heic files.
	img, format, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "unsupported or invalid image format"})
		return
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	const maxPixels = 10_000_000 // 10 million pixels
	if w*h > maxPixels {
		errorMsg := fmt.Sprintf("input image is too large: %d x %d pixels exceeds the maximum of %d pixels", w, h, maxPixels)
		// HTTP 413: Payload Too Large is the correct status code.
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": errorMsg})
		return
	}
	fmt.Printf("Successfully decoded image: %s, Format: %s, Dimensions: %d x %d\n", file.Filename, format, w, h)

	// 4. Call the selected upscaler service.
	upscaledImg, err := upscalerService.ProcessImage(img)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to upscale image: %v", err)})
		return
	}

	c.Writer.Header().Set("Content-Type", "image/png")
	if err := png.Encode(c.Writer, upscaledImg); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to encode upscaled image"})
		return
	}
}
