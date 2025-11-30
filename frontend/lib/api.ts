// Get API URL from environment or construct from current host
function getApiUrl(): string {
  // In browser, always use same host as current page but with backend port
  // This ensures it works when deployed on remote servers
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol
    const hostname = window.location.hostname
    // Backend runs on port 21115
    return `${protocol}//${hostname}:21115`
  }
  
  // For server-side rendering, check environment variable first
  // If set and not localhost, use it; otherwise fallback to localhost
  if (process.env.NEXT_PUBLIC_API_URL) {
    const envUrl = process.env.NEXT_PUBLIC_API_URL
    // If environment URL is not localhost, use it
    if (!envUrl.includes('localhost') && !envUrl.includes('127.0.0.1')) {
      return envUrl
    }
  }
  
  // Fallback for server-side rendering
  return "http://localhost:21115"
}

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

export interface ProgressUpdate {
  type: "progress" | "complete" | "error"
  step?: number
  total_steps?: number
  progress?: number
  message?: string
  image_base64?: string
  seed?: number
  generation_time_ms?: number
  width?: number
  height?: number
  image_id?: string
}

export type ProgressCallback = (update: ProgressUpdate) => void

export async function generateImage(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await fetch(`${getApiUrl()}/generate`, {
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

export async function generateImageStream(
  request: GenerateRequest,
  onProgress: ProgressCallback
): Promise<GenerateResponse> {
  return new Promise((resolve, reject) => {
    // Use fetch with ReadableStream for SSE (EventSource only supports GET)
    fetch(`${getApiUrl()}/generate-stream`, {
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
      .then(async (response) => {
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ""

        if (!reader) {
          if (!response.ok) {
            let errorMsg = `HTTP error! status: ${response.status}`
            if (response.status === 404) {
              errorMsg = `Endpoint not found. The server may need to be restarted to load the /generate-stream endpoint.`
            } else if (response.status === 405) {
              errorMsg = `Method not allowed. Please ensure the server supports POST requests to /generate-stream.`
            } else {
              // Try to read error as JSON
              try {
                const error = await response.json()
                errorMsg = error.detail || error.message || errorMsg
              } catch {
                // If not JSON, use status-based message
              }
            }
            reject(new Error(errorMsg))
          } else {
            reject(new Error("Response body is not readable"))
          }
          return
        }

        // Even if response is not ok, try to read error message from stream
        if (!response.ok) {
          // Try to read error message from stream
          try {
            const { done, value } = await reader.read()
            if (!done && value) {
              buffer += decoder.decode(value, { stream: true })
              const lines = buffer.split("\n")
              for (const line of lines) {
                if (line.startsWith("data: ")) {
                  try {
                    const data = JSON.parse(line.slice(6))
                    if (data.type === "error") {
                      reject(new Error(data.message || "Generation failed"))
                      return
                    }
                  } catch (e) {
                    // Fall through to generic error
                  }
                }
              }
            }
          } catch (e) {
            // Fall through to generic error
          }
          // Try to get more details about the error
          let errorMsg = `HTTP error! status: ${response.status}`
          if (response.status === 404) {
            errorMsg = `Endpoint not found. The server may need to be restarted to load the /generate-stream endpoint.`
          } else if (response.status === 405) {
            errorMsg = `Method not allowed. Please ensure the server supports POST requests to /generate-stream.`
          }
          reject(new Error(errorMsg))
          return
        }

        const processChunk = async () => {
          try {
            while (true) {
              const { done, value } = await reader.read()
              
              if (done) {
                break
              }

              buffer += decoder.decode(value, { stream: true })
              const lines = buffer.split("\n")
              buffer = lines.pop() || ""

              for (const line of lines) {
                if (line.startsWith("data: ")) {
                  try {
                    const data = JSON.parse(line.slice(6))
                    onProgress(data)

                    if (data.type === "complete") {
                      resolve({
                        image_base64: data.image_base64,
                        seed: data.seed,
                        generation_time_ms: data.generation_time_ms,
                        width: data.width,
                        height: data.height,
                        image_id: data.image_id,
                      })
                      return
                    } else if (data.type === "error") {
                      reject(new Error(data.message || "Generation failed"))
                      return
                    }
                  } catch (e) {
                    console.warn("Failed to parse SSE data:", e)
                  }
                }
              }
            }
          } catch (error) {
            reject(error)
          }
        }

        processChunk()
      })
      .catch(reject)
  })
}

export async function checkHealth(): Promise<boolean> {
  try {
    // Create an AbortController for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
    
    const response = await fetch(`${getApiUrl()}/health`, {
      method: "GET",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
      },
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      console.warn(`Health check failed with status ${response.status}`)
      return false
    }
    
    // Parse the response to check the actual status
    const data = await response.json()
    
    // Consider healthy if:
    // 1. Overall status is "healthy", OR
    // 2. Gateway is running AND text2image server status is "healthy", OR
    // 3. Gateway is running AND model is loaded and CUDA is available
    const isHealthy = 
      data.status === "healthy" ||
      (data.gateway === "running" && 
       (data.text2image_server?.status === "healthy" ||
        (data.text2image_server?.model_loaded === true && 
         data.text2image_server?.cuda_available === true)))
    
    // Log for debugging
    if (!isHealthy) {
      console.warn("Health check: API not healthy", {
        status: data.status,
        gateway: data.gateway,
        text2image_server: data.text2image_server,
      })
    }
    
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

