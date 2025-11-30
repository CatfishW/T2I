const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:21115"

export interface GenerateRequest {
  prompt: string
  negative_prompt?: string
  height?: number
  width?: number
  seed?: number
  num_inference_steps?: number
  guidance_scale?: number
}

export interface GenerateResponse {
  image_base64: string
  seed: number
  generation_time_ms: number
  width: number
  height: number
  image_id?: string
}

export async function generateImage(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: request.prompt,
      negative_prompt: request.negative_prompt || "",
      height: request.height || 1024,
      width: request.width || 1024,
      seed: request.seed === undefined ? -1 : request.seed,
      num_inference_steps: request.num_inference_steps || 9,
      guidance_scale: request.guidance_scale || 0.0,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

export async function checkHealth(): Promise<boolean> {
  try {
    // Create an AbortController for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
    
    const response = await fetch(`${API_URL}/health`, {
      method: "GET",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
      },
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      return false
    }
    
    // Parse the response to check the actual status
    const data = await response.json()
    
    // Only consider healthy if:
    // 1. Overall status is "healthy"
    // 2. Text2Image server has model loaded and CUDA available
    const isHealthy = 
      data.status === "healthy" &&
      data.text2image_server?.model_loaded === true &&
      data.text2image_server?.cuda_available === true
    
    return isHealthy
  } catch (error) {
    // Handle network errors, timeouts, and other failures
    if (error instanceof Error && error.name === "AbortError") {
      console.warn("Health check timed out")
    } else {
      console.warn("Health check failed:", error)
    }
    return false
  }
}

