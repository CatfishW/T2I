"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Navbar } from "@/components/navbar"
import { PrinterAnimation } from "@/components/printer-animation"
import { generateImage, generateImageStream, checkHealth, type ProgressUpdate, type GenerateResponse, type GenerateRequest } from "@/lib/api"
import { RESOLUTION_PRESETS, type GeneratedImage } from "@/types"
import { toast } from "sonner"
import {
  Loader2,
  Sparkles,
  Copy,
  Download,
  Shuffle,
  Image as ImageIcon,
  Wand2,
  Settings2,
  History,
  Maximize2,
  Minimize2,
  Share2,
  X,
  ChevronRight,
  ChevronLeft,
  Trash2,
  List,
  Clock
} from "lucide-react"

// Animation variants
// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
  exit: { opacity: 0, y: 10, transition: { duration: 0.3 } }
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.4, ease: "easeOut" as const } }
}

type Tab = "text-to-image" | "image-to-video" | "image-to-image"

export default function Home() {
  // State
  const [activeTab, setActiveTab] = useState<Tab>("text-to-image")
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [seed, setSeed] = useState<number>(-1)
  const [steps, setSteps] = useState(9)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentImage, setCurrentImage] = useState<GeneratedImage | null>(null)
  const [gallery, setGallery] = useState<GeneratedImage[]>([])
  const [apiHealth, setApiHealth] = useState<boolean | null>(null)
  const [progressValue, setProgressValue] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [queue, setQueue] = useState<GenerateRequest[]>([])

  // UI State
  const [showSettings, setShowSettings] = useState(true)
  const [showGallery, setShowGallery] = useState(false)
  const [activeSidebarTab, setActiveSidebarTab] = useState<"history" | "queue">("history")
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Refs
  const promptInputRef = useRef<HTMLTextAreaElement>(null)

  // Check API health
  useEffect(() => {
    const performHealthCheck = async () => {
      try {
        const isHealthy = await checkHealth()
        setApiHealth(isHealthy)
      } catch (error) {
        console.error("Health check error:", error)
        setApiHealth(false)
      }
    }
    performHealthCheck()
    const interval = setInterval(performHealthCheck, 30000)
    return () => clearInterval(interval)
  }, [])

  // Reset progress
  useEffect(() => {
    if (!isGenerating) {
      setProgressValue(0)
      setCurrentStep(0)
      setTotalSteps(0)
    }
  }, [isGenerating])

  const processQueueItem = useCallback(async (request: GenerateRequest) => {
    setIsGenerating(true)
    setIsAnimating(true)
    setCurrentImage(null)
    setProgressValue(0)
    setCurrentStep(0)
    // Update prompt state for animation
    setPrompt(request.prompt)

    try {
      const startTime = Date.now()
      let response: GenerateResponse

      try {
        response = await generateImageStream(
          request,
          (update: ProgressUpdate) => {
            if (update.type === "progress") {
              setCurrentStep(update.step || 0)
              setTotalSteps(update.total_steps || (request.num_inference_steps || steps))
              setProgressValue(Math.min(100, Math.max(0, update.progress || 0)))
            } else if (update.type === "complete") {
              setProgressValue(100)
              setCurrentStep(request.num_inference_steps || steps)
              setTotalSteps(request.num_inference_steps || steps)
            }
          }
        )
      } catch (streamError) {
        // Fallback logic
        if (streamError instanceof Error && streamError.message.includes("404")) {
          console.warn("Streaming endpoint not available, using regular endpoint")
          const progressInterval = setInterval(() => {
            setProgressValue((prev) => Math.min(prev + Math.random() * 10, 90))
          }, 300)

          response = await generateImage(request)

          clearInterval(progressInterval)
          setProgressValue(100)
          setCurrentStep(request.num_inference_steps || steps)
          setTotalSteps(request.num_inference_steps || steps)
        } else {
          throw streamError
        }
      }

      // Validate image
      let imageBase64 = response.image_base64
      if (!imageBase64) throw new Error("No image data received")
      if (!imageBase64.startsWith("data:image/")) {
        if (imageBase64 && !imageBase64.includes(",")) {
          imageBase64 = `data:image/png;base64,${imageBase64}`
        }
      }

      const generatedImage: GeneratedImage = {
        id: response.image_id || `img-${Date.now()}`,
        imageBase64: imageBase64,
        prompt: request.prompt,
        negativePrompt: request.negative_prompt || "",
        seed: response.seed,
        width: response.width,
        height: response.height,
        steps: request.num_inference_steps || steps,
        generationTimeMs: response.generation_time_ms,
        timestamp: startTime,
      }

      setCurrentImage(generatedImage)
      setGallery((prev) => [generatedImage, ...prev].slice(0, 50))
      toast.success(`✨ Image generated in ${response.generation_time_ms}ms`)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to generate image"
      toast.error(message)
    } finally {
      setIsGenerating(false)
    }
  }, [steps])

  // Queue processing effect
  useEffect(() => {
    if (!isGenerating && queue.length > 0) {
      const nextRequest = queue[0]
      setQueue((prev) => prev.slice(1))
      processQueueItem(nextRequest)
    }
  }, [isGenerating, queue, processQueueItem])

  const handleGenerate = useCallback(() => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt")
      promptInputRef.current?.focus()
      return
    }

    if (apiHealth === false) {
      toast.error("API is not available. Please check the backend connection.")
      return
    }

    const request: GenerateRequest = {
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      seed: seed >= 0 ? seed : -1,
      num_inference_steps: steps,
    }

    setQueue((prev) => [...prev, request])
    if (isGenerating || queue.length > 0) {
      toast.success("Added to queue")
      setShowGallery(true)
      setActiveSidebarTab("queue")
    }
  }, [prompt, negativePrompt, width, height, seed, steps, apiHealth, isGenerating, queue.length])

  const removeFromQueue = (index: number) => {
    setQueue((prev) => prev.filter((_, i) => i !== index))
    toast.success("Removed from queue")
  }

  const handleRandomSeed = () => {
    const newSeed = Math.floor(Math.random() * 2 ** 32)
    setSeed(newSeed)
    toast.success(`🎲 Random seed: ${newSeed}`)
  }

  const handleDownload = (image: GeneratedImage) => {
    const link = document.createElement("a")
    link.href = image.imageBase64
    link.download = `rowan-ai-${image.seed}-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    toast.success("💾 Image downloaded")
  }

  const handlePresetSelect = (preset: typeof RESOLUTION_PRESETS[0]) => {
    setWidth(preset.width)
    setHeight(preset.height)
  }

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden flex flex-col">
      {/* Background Ambient Effects */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/5 blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-secondary/5 blur-[120px]" />
      </div>

      <Navbar activeTab={activeTab} onTabChange={setActiveTab} apiHealth={apiHealth} />

      <main className="flex-1 relative z-10 flex h-[calc(100vh-4rem)]">

        {/* Left Sidebar - Settings */}
        <AnimatePresence mode="wait">
          {showSettings && (
            <motion.aside
              initial={{ x: -320, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -320, opacity: 0 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="w-80 glass-panel h-full overflow-y-auto hidden md:block"
            >
              <div className="p-6 space-y-8">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Settings2 className="w-5 h-5 text-primary" />
                    Configuration
                  </h2>
                </div>

                {/* Dimensions */}
                <div className="space-y-4">
                  <Label className="text-sm font-medium text-muted-foreground">Dimensions</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {RESOLUTION_PRESETS.map((preset) => (
                      <button
                        key={preset.label}
                        onClick={() => handlePresetSelect(preset)}
                        className={`p-2 rounded-lg border text-xs transition-all flex flex-col items-center gap-2 relative group ${width === preset.width && height === preset.height
                          ? "bg-primary/10 border-primary text-primary font-medium"
                          : "bg-muted/50 border-transparent hover:bg-muted text-muted-foreground"
                          }`}
                        title={preset.hint}
                      >
                        <div className="aspect-square w-full bg-current opacity-20 rounded-sm"
                          style={{ aspectRatio: `${preset.width}/${preset.height}` }} />
                        <span>{preset.label}</span>

                        {/* Tooltip Hint */}
                        <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-[10px] px-2 py-1 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-[100]">
                          {preset.hint}
                          <div className="absolute bottom-[-4px] left-1/2 -translate-x-1/2 border-4 border-transparent border-t-popover" />
                        </div>
                      </button>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <Label className="text-xs text-muted-foreground">Width</Label>
                      <Input
                        type="number"
                        value={width}
                        min={256}
                        max={2048}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setWidth(val)
                        }}
                        onBlur={() => setWidth(Math.max(256, Math.min(2048, width)))}
                        className="h-8 bg-muted/50 border-transparent"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs text-muted-foreground">Height</Label>
                      <Input
                        type="number"
                        value={height}
                        min={256}
                        max={2048}
                        onChange={(e) => {
                          const val = Number(e.target.value)
                          setHeight(val)
                        }}
                        onBlur={() => setHeight(Math.max(256, Math.min(2048, height)))}
                        className="h-8 bg-muted/50 border-transparent"
                      />
                    </div>
                  </div>
                </div>

                {/* Steps */}
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label className="text-sm font-medium text-muted-foreground">Quality Steps</Label>
                    <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded">{steps}</span>
                  </div>
                  <Slider
                    value={[steps]}
                    onValueChange={(vals) => setSteps(vals[0])}
                    min={6}
                    max={12}
                    step={1}
                    className="py-2"
                  />
                </div>

                {/* Seed */}
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-medium text-muted-foreground">Seed</Label>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={handleRandomSeed}
                      title="Randomize Seed"
                    >
                      <Shuffle className="w-3 h-3" />
                    </Button>
                  </div>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      value={seed === -1 ? "" : seed}
                      placeholder="Random (-1)"
                      onChange={(e) => setSeed(e.target.value === "" ? -1 : Number(e.target.value))}
                      className="h-9 bg-muted/50 border-transparent font-mono text-xs"
                    />
                  </div>
                </div>

                {/* Negative Prompt */}
                <div className="space-y-4">
                  <Label className="text-sm font-medium text-muted-foreground">Negative Prompt</Label>
                  <Textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="What to avoid..."
                    className="min-h-[80px] bg-muted/50 border-transparent resize-none text-sm"
                    maxLength={2000}
                  />
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Stage */}
        <div className="flex-1 flex flex-col relative overflow-hidden">
          {/* Toggle Sidebar Button */}
          <Button
            variant="ghost"
            size="icon"
            className="absolute top-4 left-4 z-20 md:hidden"
            onClick={() => setShowSettings(!showSettings)}
          >
            {showSettings ? <X className="w-5 h-5" /> : <Settings2 className="w-5 h-5" />}
          </Button>

          {/* Image Display Area */}
          <div className="flex-1 flex items-center justify-center p-4 sm:p-8 lg:p-12 relative">
            <AnimatePresence mode="wait">
              {isAnimating ? (
                <motion.div
                  key="printer"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="w-full h-full flex items-center justify-center"
                >
                  <PrinterAnimation
                    prompt={prompt}
                    image={currentImage?.imageBase64 || null}
                    onComplete={() => setIsAnimating(false)}
                  />
                </motion.div>
              ) : currentImage ? (
                <motion.div
                  key={currentImage.id}
                  variants={scaleIn}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  className={`relative group max-w-full max-h-full shadow-2xl rounded-lg overflow-hidden ${isFullscreen ? "fixed inset-0 z-50 bg-background/95 flex items-center justify-center" : ""
                    }`}
                >
                  <img
                    src={currentImage.imageBase64}
                    alt={currentImage.prompt}
                    className={`object-contain ${isFullscreen ? "max-h-screen w-auto" : "max-h-[calc(100vh-16rem)] w-auto"}`}
                  />

                  {/* Image Overlays */}
                  <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <Button size="icon" variant="secondary" className="h-8 w-8 backdrop-blur-md bg-background/50" onClick={() => setIsFullscreen(!isFullscreen)}>
                      {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                    </Button>
                    <Button size="icon" variant="secondary" className="h-8 w-8 backdrop-blur-md bg-background/50" onClick={() => handleDownload(currentImage)}>
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center space-y-6 max-w-md mx-auto"
                >
                  <div className="w-24 h-24 rounded-3xl bg-muted/30 mx-auto flex items-center justify-center animate-float-slow">
                    <ImageIcon className="w-10 h-10 text-muted-foreground/50" />
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-2xl font-bold text-gradient-brown">Ready to Create</h3>
                    <p className="text-muted-foreground">
                      Enter a prompt below to generate stunning AI art using our advanced models.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Bottom Floating Bar */}
          <div className="w-full max-w-3xl mx-auto p-4 pb-8 relative z-20">
            <motion.div
              className="glass rounded-2xl p-2 shadow-2xl ring-1 ring-white/20"
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <div className="relative">
                <Textarea
                  ref={promptInputRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value.slice(0, 2000))}
                  placeholder="Describe your imagination..."
                  className="min-h-[60px] max-h-[120px] pr-[250px] bg-transparent border-none focus-visible:ring-0 resize-none text-base py-3 px-4"
                  maxLength={2000}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleGenerate()
                    }
                  }}
                />
                <div className="absolute bottom-2 right-2 flex items-center gap-2">
                  <span className="text-xs text-muted-foreground hidden sm:inline-block">
                    {prompt.length}/2000
                  </span>
                  <Button
                    onClick={handleGenerate}
                    disabled={!prompt.trim()}
                    size="sm"
                    className="h-9 px-4 bg-gradient-to-r from-primary to-rowan-gold hover:opacity-90 transition-opacity text-white shadow-lg"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        {queue.length > 0 ? `Queue (${queue.length})` : "Generating..."}
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 mr-2" />
                        {queue.length > 0 ? `Queue (${queue.length})` : "Generate"}
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Progress Bar */}
              {isGenerating && (
                <div className="absolute -bottom-1 left-4 right-4 h-1 bg-muted overflow-hidden rounded-full">
                  <motion.div
                    className="h-full bg-gradient-to-r from-primary to-rowan-gold"
                    initial={{ width: 0 }}
                    animate={{ width: `${progressValue}%` }}
                    transition={{ ease: "linear" }}
                  />
                </div>
              )}
            </motion.div>
          </div>
        </div >

        {/* Right Sidebar - Gallery (Collapsible) */}
        <AnimatePresence>
          {
            showGallery && (
              <motion.aside
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 320, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                className="border-l border-border bg-card/50 backdrop-blur-sm hidden md:flex flex-col z-20 absolute right-0 top-0 bottom-0 shadow-2xl"
              >
                <div className="p-3 border-b border-border">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold flex items-center gap-2">
                      <History className="w-4 h-4" />
                      Activity
                    </h3>
                    <Button variant="ghost" size="icon" onClick={() => setShowGallery(false)} className="h-8 w-8">
                      <X className="w-4 h-4" />
                    </Button>
                  </div>

                  <div className="flex p-1 bg-muted/50 rounded-lg">
                    <button
                      onClick={() => setActiveSidebarTab("queue")}
                      className={`flex-1 flex items-center justify-center gap-2 text-xs font-medium py-1.5 rounded-md transition-all ${activeSidebarTab === "queue"
                        ? "bg-background shadow-sm text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      <List className="w-3 h-3" />
                      Queue ({queue.length})
                    </button>
                    <button
                      onClick={() => setActiveSidebarTab("history")}
                      className={`flex-1 flex items-center justify-center gap-2 text-xs font-medium py-1.5 rounded-md transition-all ${activeSidebarTab === "history"
                        ? "bg-background shadow-sm text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      <Clock className="w-3 h-3" />
                      History ({gallery.length})
                    </button>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  {activeSidebarTab === "queue" ? (
                    <div className="space-y-3">
                      {queue.length === 0 && (
                        <div className="text-center text-muted-foreground py-8 text-sm flex flex-col items-center gap-2">
                          <List className="w-8 h-8 opacity-20" />
                          <p>Queue is empty</p>
                        </div>
                      )}
                      {queue.map((req, i) => (
                        <motion.div
                          key={`queue-${i}`}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          className="bg-muted/30 rounded-lg p-3 border border-border flex gap-3 group relative overflow-hidden"
                        >
                          <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/20" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate pr-6" title={req.prompt}>{req.prompt}</p>
                            <div className="flex items-center gap-2 mt-1.5 text-xs text-muted-foreground">
                              <span className="bg-primary/10 text-primary px-1.5 py-0.5 rounded flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                Waiting
                              </span>
                              <span>{req.width}x{req.height}</span>
                              <span>{req.num_inference_steps} steps</span>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive hover:bg-destructive/10 absolute right-2 top-2"
                            onClick={() => removeFromQueue(i)}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {gallery.map((img) => (
                        <motion.div
                          key={img.id}
                          layoutId={img.id}
                          className="group relative aspect-square rounded-lg overflow-hidden cursor-pointer ring-1 ring-border hover:ring-primary transition-all"
                          onClick={() => setCurrentImage(img)}
                        >
                          <img src={img.imageBase64} alt={img.prompt} className="w-full h-full object-cover" />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                            <Button size="icon" variant="secondary" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); handleDownload(img) }}>
                              <Download className="w-4 h-4" />
                            </Button>
                          </div>
                          <div className="absolute bottom-1 right-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded backdrop-blur-sm pointer-events-none">
                            {(img.generationTimeMs / 1000).toFixed(1)}s
                          </div>
                        </motion.div>
                      ))}
                      {gallery.length === 0 && (
                        <div className="text-center text-muted-foreground py-8 text-sm flex flex-col items-center gap-2">
                          <History className="w-8 h-8 opacity-20" />
                          <p>No images generated yet.</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </motion.aside>
            )
          }
        </AnimatePresence >

        {/* Gallery Toggle (if hidden) */}
        {
          !showGallery && (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 z-20 hidden md:block">
              <Button
                variant="secondary"
                size="sm"
                className="h-16 w-6 rounded-l-lg rounded-r-none p-0 shadow-lg border-l border-y border-border"
                onClick={() => setShowGallery(true)}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
            </div>
          )
        }
      </main >
    </div >
  )
}
