package api

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// IPRateLimiter holds the rate limiters for each IP address.
type IPRateLimiter struct {
	mu       sync.Mutex
	limiters map[string]*rate.Limiter
}

// NewIPRateLimiter creates a new rate limiter.
// 'r' is the number of events per second, 'b' is the burst size.
// For example, 1 request every 10 seconds is rate.Every(10 * time.Second)
func NewIPRateLimiter(r rate.Limit, b int) *IPRateLimiter {
	return &IPRateLimiter{
		limiters: make(map[string]*rate.Limiter),
	}
}

// getLimiter retrieves or creates a limiter for a given IP.
func (i *IPRateLimiter) getLimiter(ip string) *rate.Limiter {
	i.mu.Lock()
	defer i.mu.Unlock()

	limiter, exists := i.limiters[ip]
	if !exists {
		// Create a new limiter for this IP.
		// Allows 20 requests with a burst of 20, effectively a daily limit
		// that resets. A more robust solution would use a timed cache.
		// For simplicity, we allow a burst of 20 requests and then block.
		limiter = rate.NewLimiter(rate.Every(24*time.Hour/20), 20) // 20 requests per 24 hours
		i.limiters[ip] = limiter
	}
	return limiter
}

// RateLimitMiddleware is the Gin middleware function.
func RateLimitMiddleware() gin.HandlerFunc {
	// For this simple implementation, we'll create the limiter here.
	// A more advanced version would pass this in from main.go.
	limiter := NewIPRateLimiter(rate.Every(time.Minute), 100) // General settings, specific logic is in getLimiter

	return func(c *gin.Context) {
		ip := c.ClientIP()
		ipLimiter := limiter.getLimiter(ip)

		if !ipLimiter.Allow() {
			// If the request is not allowed, send a 429 Too Many Requests error.
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "too many requests"})
			return
		}
		// If allowed, proceed to the next handler.
		c.Next()
	}
}
