package api

import (
	"net/http"

	"github.com/azadraja/image-upscale-api/internal/upscaler"
	"github.com/gin-gonic/gin"
)

// SetupRouter configures all the routes for the application.
func SetupRouter(upscalerServices map[string]map[int]*upscaler.Upscaler) *gin.Engine {
	// Initialize a new Gin router with default middleware.
	router := gin.Default()

	// Use a middleware to make the upscaler services map available to all handlers.
	router.Use(func(c *gin.Context) {
		c.Set("upscalers", upscalerServices)
		c.Next()
	})

	// Group API routes under /api/v1
	apiV1 := router.Group("/api/v1")
	{
		// Connect the POST /upscale endpoint to the HandleUpscale function.
		apiV1.POST("/upscale", RateLimitMiddleware(), HandleUpscale)
	}

	// Health check endpoint remains for simple verification.
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	return router
}
