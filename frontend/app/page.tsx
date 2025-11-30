"use client"

import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { generateImage, generateImageStream, checkHealth, type ProgressUpdate } from "@/lib/api"
import { RESOLUTION_PRESETS, type GeneratedImage } from "@/types"
import { toast } from "sonner"
import {
  Loader2,
  Sparkles,
  Copy,
  Download,
  ChevronDown,
  ChevronUp,
  Shuffle,
  Image as ImageIcon,
  AlertCircle,
  Zap,
  Wand2,
  TrendingUp,
  Clock,
  Layers,
  Palette,
  Info,
  Lightbulb,
  Star,
  Video,
} from "lucide-react"

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const cardVariants = {
  hidden: { opacity: 0, scale: 0.95, y: 20 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const imageVariants = {
  hidden: { opacity: 0, scale: 0.8, rotate: -5 },
  visible: {
    opacity: 1,
    scale: 1,
    rotate: 0,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

const galleryItemVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: (i: number) => ({
    opacity: 1,
    scale: 1,
    transition: {
      delay: i * 0.05,
      duration: 0.3,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  }),
  hover: {
    scale: 1.05,
    rotate: 2,
    transition: {
      duration: 0.2,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
}

type Tab = "text-to-image" | "image-to-video" | "image-to-image"

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("text-to-image")
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("")
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [seed, setSeed] = useState<number>(-1)
  const [steps, setSteps] = useState(9)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<GeneratedImage | null>(null)
  const [gallery, setGallery] = useState<GeneratedImage[]>([])
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null)
  const [apiHealth, setApiHealth] = useState<boolean | null>(null)
  const [progressValue, setProgressValue] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [showTips, setShowTips] = useState(true)
  const [showLeftPanel, setShowLeftPanel] = useState(false)
  const [showRightPanel, setShowRightPanel] = useState(false)
  const [isMobile, setIsMobile] = useState(true) // Default to true to hide panels on initial render

  // Check if mobile on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
    }
    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  // Check API health on mount and periodically
  useEffect(() => {
    // Check immediately on mount
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
    
    // Check every 30 seconds
    const interval = setInterval(performHealthCheck, 30000)
    return () => clearInterval(interval)
  }, [])

  // Reset progress when generation starts/stops
  useEffect(() => {
    if (!isGenerating) {
      setProgressValue(0)
      setCurrentStep(0)
      setTotalSteps(0)
    }
  }, [isGenerating])

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt")
      return
    }

    if (apiHealth === false) {
      toast.error("API is not available. Please check the backend connection.")
      return
    }

    setIsGenerating(true)
    setProgressValue(0)
    setCurrentStep(0)
    setTotalSteps(steps)
    
    try {
      const startTime = Date.now()
      
      const response = await generateImageStream(
        {
          prompt,
          negative_prompt: negativePrompt,
          width,
          height,
          seed: seed >= 0 ? seed : undefined,
          num_inference_steps: steps,
        },
        (update: ProgressUpdate) => {
          if (update.type === "progress") {
            setCurrentStep(update.step || 0)
            setTotalSteps(update.total_steps || steps)
            setProgressValue(update.progress || 0)
          } else if (update.type === "complete") {
            setProgressValue(100)
            setCurrentStep(update.total_steps || steps)
          }
        }
      )

      const generatedImage: GeneratedImage = {
        id: response.image_id || `img-${Date.now()}`,
        imageBase64: response.image_base64,
        prompt,
        negativePrompt,
        seed: response.seed,
        width: response.width,
        height: response.height,
        steps,
        generationTimeMs: response.generation_time_ms,
        timestamp: startTime,
      }

      setCurrentImage(generatedImage)
      setGallery((prev) => [generatedImage, ...prev].slice(0, 12))
      toast.success(`✨ Image generated in ${response.generation_time_ms}ms`)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to generate image"
      toast.error(message)
      console.error("Generation error:", error)
    } finally {
      setIsGenerating(false)
    }
  }, [prompt, negativePrompt, width, height, seed, steps, apiHealth])

  const handleRandomSeed = () => {
    const newSeed = Math.floor(Math.random() * 2 ** 32)
    setSeed(newSeed)
    toast.success(`🎲 Random seed: ${newSeed}`)
  }

  const handleCopyPrompt = () => {
    navigator.clipboard.writeText(prompt)
    toast.success("📋 Prompt copied to clipboard")
  }

  const handleCopySeed = () => {
    if (currentImage) {
      navigator.clipboard.writeText(currentImage.seed.toString())
      toast.success("📋 Seed copied to clipboard")
    }
  }

  const handleDownload = (image: GeneratedImage) => {
    const link = document.createElement("a")
    link.href = image.imageBase64
    link.download = `image-${image.seed}-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    toast.success("💾 Image downloaded")
  }

  const handlePresetSelect = (preset: typeof RESOLUTION_PRESETS[0]) => {
    setWidth(preset.width)
    setHeight(preset.height)
    toast.success(`📐 Resolution set to ${preset.label}`)
  }

  const examplePrompts = [
    "A futuristic cityscape at sunset with flying cars",
    "A serene Japanese garden with cherry blossoms",
    "A cyberpunk street with neon lights and rain",
    "A magical forest with glowing mushrooms",
    "An astronaut floating in space",
  ]

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} apiHealth={apiHealth} />

      <motion.div
        className="flex-1 relative overflow-hidden"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        {/* Rowan University themed background gradient */}
        <div className="fixed inset-0 -z-10">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/3 via-background to-secondary/3" />
          <motion.div
            className="absolute inset-0 opacity-20"
            animate={{
              background: [
                "radial-gradient(circle at 20% 50%, rgba(122, 69, 41, 0.08) 0%, transparent 50%)",
                "radial-gradient(circle at 80% 80%, rgba(204, 153, 0, 0.08) 0%, transparent 50%)",
                "radial-gradient(circle at 40% 20%, rgba(122, 69, 41, 0.06) 0%, transparent 50%)",
                "radial-gradient(circle at 60% 40%, rgba(204, 153, 0, 0.06) 0%, transparent 50%)",
                "radial-gradient(circle at 20% 50%, rgba(122, 69, 41, 0.08) 0%, transparent 50%)",
              ],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </div>

        {/* Stats Bar - Horizontal, Transparent, Centered */}
        <motion.div
          className="w-full bg-background/40 backdrop-blur-sm border-b border-border/50"
          variants={itemVariants}
        >
          <div className="container mx-auto px-2 sm:px-4 max-w-7xl">
            <div className="flex items-center justify-center gap-2 sm:gap-4 md:gap-8 py-2 sm:py-3 flex-wrap">
              <div className="flex items-center gap-1 sm:gap-2">
                <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                <span className="text-[10px] sm:text-xs text-muted-foreground font-medium">Total:</span>
                <span className="text-xs sm:text-sm font-bold text-foreground">{gallery.length}</span>
              </div>
              <div className="flex items-center gap-1 sm:gap-2">
                <Clock className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                <span className="text-[10px] sm:text-xs text-muted-foreground font-medium">Time:</span>
                <span className="text-xs sm:text-sm font-bold text-foreground">
                  {gallery.length > 0
                    ? Math.round(
                        gallery.reduce((acc, img) => acc + img.generationTimeMs, 0) /
                          gallery.length
                      )
                    : 0}
                  ms
                </span>
              </div>
              <div className="flex items-center gap-1 sm:gap-2">
                <Layers className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                <span className="text-[10px] sm:text-xs text-muted-foreground font-medium">Res:</span>
                <span className="text-xs sm:text-sm font-bold text-primary">
                  {width}×{height}
                </span>
              </div>
              <div className="flex items-center gap-1 sm:gap-2">
                <Palette className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                <span className="text-[10px] sm:text-xs text-muted-foreground font-medium">Steps:</span>
                <span className="text-xs sm:text-sm font-bold text-foreground">{steps}</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* 3-Column Pro Studio Layout */}
        <div className="flex flex-col lg:flex-row h-[calc(100vh-3.5rem)] sm:h-[calc(100vh-4rem)] relative z-10">
          {/* Mobile Panel Toggle Buttons */}
          <div className="lg:hidden flex gap-2 p-2 border-b border-border/50 bg-background/40 backdrop-blur-sm">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowLeftPanel(!showLeftPanel)}
              className="flex-1"
            >
              <Wand2 className="h-4 w-4 mr-2" />
              Settings
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowRightPanel(!showRightPanel)}
              className="flex-1"
            >
              <Lightbulb className="h-4 w-4 mr-2" />
              Tips
            </Button>
          </div>

          {/* Left Panel - Technical Controls (280px fixed on desktop, collapsible on mobile) */}
          <AnimatePresence>
            {(showLeftPanel || !isMobile) && (
              <motion.div
                className={`${
                  showLeftPanel || !isMobile ? "block" : "hidden"
                } w-full lg:w-[280px] bg-card/50 backdrop-blur-sm border-r border-border/50 overflow-y-auto max-h-[50vh] lg:max-h-none`}
                variants={itemVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
              >
            <div className="p-4 sm:p-6 space-y-4 sm:space-y-6">
              {/* Generation Settings Title */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Wand2 className="h-5 w-5 text-primary" />
                  <h2 className="text-base sm:text-lg font-semibold">Generation Settings</h2>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowLeftPanel(false)}
                  className="lg:hidden"
                >
                  ×
                </Button>
              </div>

              {/* Resolution - Icon Grid */}
              <motion.div className="space-y-3" variants={itemVariants}>
                <Label className="text-sm font-medium">Resolution</Label>
                <div className="grid grid-cols-3 gap-2">
                  {RESOLUTION_PRESETS.map((preset, index) => {
                    const isActive = width === preset.width && height === preset.height
                    const isSquare = preset.width === preset.height
                    const isPortrait = preset.height > preset.width
                    const isLandscape = preset.width > preset.height
                    
                    return (
                      <motion.div
                        key={preset.label}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <button
                          onClick={() => handlePresetSelect(preset)}
                          className={`w-full aspect-square rounded-lg p-2 flex items-center justify-center transition-all min-h-[44px] touch-manipulation ${
                            isActive
                              ? "bg-primary text-primary-foreground shadow-md"
                              : "bg-muted hover:bg-muted/80 border border-border/50"
                          }`}
                          title={preset.label}
                        >
                          {isSquare ? (
                            <div className="w-6 h-6 border-2 border-current rounded" />
                          ) : isPortrait ? (
                            <div className="w-4 h-6 border-2 border-current rounded" />
                          ) : (
                            <div className="w-6 h-4 border-2 border-current rounded" />
                          )}
                        </button>
                      </motion.div>
                    )
                  })}
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="space-y-1">
                    <Label htmlFor="width" className="text-xs">Width</Label>
                    <Input
                      id="width"
                      type="number"
                      value={width}
                      onChange={(e) => setWidth(Math.max(256, Math.min(2048, parseInt(e.target.value) || 1024)))}
                      min={256}
                      max={2048}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="height" className="text-xs">Height</Label>
                    <Input
                      id="height"
                      type="number"
                      value={height}
                      onChange={(e) => setHeight(Math.max(256, Math.min(2048, parseInt(e.target.value) || 1024)))}
                      min={256}
                      max={2048}
                      className="h-8 text-xs"
                    />
                  </div>
                </div>
              </motion.div>

              {/* Seed */}
              <motion.div className="space-y-2" variants={itemVariants}>
                <div className="flex items-center justify-between">
                  <Label htmlFor="seed" className="text-sm font-medium">Seed</Label>
                  <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9, rotate: 180 }}>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomSeed}
                      className="h-7 text-xs"
                    >
                      <Shuffle className="h-3 w-3 mr-1" />
                      Random
                    </Button>
                  </motion.div>
                </div>
                <Input
                  id="seed"
                  type="number"
                  placeholder="Random"
                  value={seed >= 0 ? seed : ""}
                  onChange={(e) => {
                    const val = e.target.value
                    setSeed(val === "" ? -1 : parseInt(val) || -1)
                  }}
                  min={0}
                  max={2 ** 32 - 1}
                  className="h-8 text-xs"
                />
              </motion.div>

              {/* Steps */}
              <motion.div className="space-y-2" variants={itemVariants}>
                <div className="flex items-center justify-between">
                  <Label htmlFor="steps" className="text-sm font-medium">Steps: {steps}</Label>
                  <motion.span
                    key={steps}
                    initial={{ scale: 1.2, color: "var(--primary)" }}
                    animate={{ scale: 1, color: "var(--foreground)" }}
                    className="text-sm font-semibold"
                  >
                    {steps}
                  </motion.span>
                </div>
                <Slider
                  id="steps"
                  min={6}
                  max={12}
                  step={1}
                  value={[steps]}
                  onValueChange={(value) => setSteps(value[0])}
                  className="cursor-pointer"
                />
              </motion.div>
            </div>
          </motion.div>
            )}
          </AnimatePresence>

          {/* Center Panel - The Stage (Flexible) */}
          <motion.div
            className="flex-1 flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 overflow-y-auto min-h-0"
            variants={itemVariants}
            initial="hidden"
            animate="visible"
          >
            <AnimatePresence mode="wait">
              {activeTab === "text-to-image" && (
                <motion.div
                  key="text-to-image"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="w-full max-w-4xl space-y-4 sm:space-y-6"
                >
                  {/* Image Display */}
                  <AnimatePresence mode="wait">
                    {currentImage ? (
                      <motion.div
                        key={currentImage.id}
                        variants={imageVariants}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        className="relative w-full aspect-square bg-muted/30 rounded-lg sm:rounded-xl overflow-hidden group"
                      >
                        <motion.img
                          src={currentImage.imageBase64}
                          alt={currentImage.prompt}
                          className="w-full h-full object-contain"
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.5, delay: 0.2 }}
                        />
                        <motion.div
                          className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity"
                          initial={false}
                        >
                          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={handleCopySeed}
                              className="bg-background/90 backdrop-blur-sm"
                            >
                              <Copy className="h-4 w-4 mr-2" />
                              Copy Seed
                            </Button>
                          </motion.div>
                          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDownload(currentImage)}
                              className="bg-background/90 backdrop-blur-sm"
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Download
                            </Button>
                          </motion.div>
                        </motion.div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="empty"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ duration: 0.4 }}
                        className="relative w-full aspect-square bg-muted/30 rounded-lg sm:rounded-xl flex items-center justify-center"
                      >
                        <motion.div
                          animate={{
                            y: [0, -10, 0],
                            rotate: [0, 5, -5, 0],
                          }}
                          transition={{
                            duration: 3,
                            repeat: Infinity,
                            ease: "easeInOut",
                          }}
                        >
                          <div className="relative">
                            <ImageIcon className="h-20 w-20 text-muted-foreground mb-4" />
                            <motion.div
                              className="absolute -inset-4 bg-primary/10 rounded-full blur-xl"
                              animate={{
                                scale: [1, 1.2, 1],
                                opacity: [0.5, 0.8, 0.5],
                              }}
                              transition={{
                                duration: 3,
                                repeat: Infinity,
                                ease: "easeInOut",
                              }}
                            />
                          </div>
                        </motion.div>
                        <p className="text-muted-foreground text-center text-sm sm:text-lg font-medium absolute bottom-4 sm:bottom-8 px-4">
                          Enter a prompt and click Generate to create an image ✨
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Prompt Input */}
                  <motion.div
                    className="space-y-2 sm:space-y-3"
                    whileHover={{ scale: !isMobile ? 1.01 : 1 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <div className="flex items-center justify-between">
                      <Label htmlFor="prompt" className="text-xs sm:text-sm font-medium">Prompt</Label>
                      <motion.span
                        className="text-[10px] sm:text-xs text-muted-foreground"
                        key={prompt.length}
                        initial={{ scale: 1.2, color: "var(--primary)" }}
                        animate={{ scale: 1, color: "var(--muted-foreground)" }}
                        transition={{ duration: 0.3 }}
                      >
                        {prompt.length}/2000
                      </motion.span>
                    </div>
                    <Textarea
                      id="prompt"
                      placeholder="Enter your prompt here... ✨"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value.slice(0, 2000))}
                      className="min-h-[100px] sm:min-h-[120px] resize-none transition-all focus:ring-2 focus:ring-primary/50 rounded-lg text-sm"
                      maxLength={2000}
                    />
                    <div className="flex gap-2">
                      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleCopyPrompt}
                          disabled={!prompt.trim()}
                          className="w-full"
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy Prompt
                        </Button>
                      </motion.div>
                    </div>
                  </motion.div>

                  {/* Generate Button - Rowan Gold */}
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="flex justify-center"
                  >
                    <Button
                      onClick={handleGenerate}
                      disabled={isGenerating || !prompt.trim() || apiHealth === false}
                      className="w-full max-w-md bg-[#CC9900] hover:bg-[#B8860B] text-primary-foreground font-semibold shadow-lg hover:shadow-xl transition-all rounded-lg min-h-[44px] touch-manipulation"
                      size="lg"
                    >
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                        animate={{
                          x: isGenerating ? ["-100%", "100%"] : "-100%",
                        }}
                        transition={{
                          duration: 2,
                          repeat: isGenerating ? Infinity : 0,
                          ease: "linear",
                        }}
                      />
                      {isGenerating ? (
                        <>
                          <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-5 w-5 mr-2" />
                          Generate Image
                        </>
                      )}
                    </Button>
                  </motion.div>
                  {/* Progress bar with step info */}
                  <AnimatePresence>
                    {isGenerating && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="w-full max-w-md mx-auto space-y-2"
                      >
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>
                            {currentStep > 0 && totalSteps > 0
                              ? `Step ${currentStep} of ${totalSteps}`
                              : "Initializing..."}
                          </span>
                          <span>{progressValue}%</span>
                        </div>
                        <div className="w-full bg-muted rounded-full overflow-hidden h-2">
                          <motion.div
                            className="h-full bg-[#CC9900]"
                            initial={{ width: 0 }}
                            animate={{ width: `${progressValue}%` }}
                            transition={{ type: "spring", stiffness: 100, damping: 30 }}
                          />
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Gallery */}
                  <AnimatePresence>
                    {gallery.length > 0 && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.4 }}
                        className="mt-8"
                      >
                        <h3 className="text-xs sm:text-sm font-semibold mb-3 sm:mb-4 text-center">Recent Images</h3>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
                          {gallery.map((img, index) => (
                            <motion.div
                              key={img.id}
                              custom={index}
                              variants={galleryItemVariants}
                              initial="hidden"
                              animate="visible"
                              whileHover="hover"
                              className="relative aspect-square bg-muted/30 rounded-md sm:rounded-lg overflow-hidden cursor-pointer group"
                              onClick={() => setSelectedImage(img)}
                            >
                              <motion.img
                                src={img.imageBase64}
                                alt={img.prompt}
                                className="w-full h-full object-cover"
                                whileHover={{ scale: 1.1 }}
                                transition={{ duration: 0.3 }}
                              />
                              <motion.div
                                className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center"
                                initial={false}
                              >
                                <motion.div
                                  initial={{ opacity: 0, scale: 0.5 }}
                                  whileHover={{ opacity: 1, scale: 1 }}
                                  transition={{ duration: 0.2 }}
                                >
                                  <ImageIcon className="h-6 w-6 text-white" />
                                </motion.div>
                              </motion.div>
                            </motion.div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              {/* Placeholder tabs */}
              {(activeTab === "image-to-video" || activeTab === "image-to-image") && (
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="flex items-center justify-center min-h-[60vh]"
                >
                  <div className="bg-card/50 backdrop-blur-sm rounded-xl p-8 max-w-md border border-border/50">
                    <motion.div
                      animate={{
                        y: [0, -10, 0],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    >
                      <div className="relative">
                        {activeTab === "image-to-video" ? (
                          <Video className="h-16 w-16 text-muted-foreground mb-4 mx-auto" />
                        ) : (
                          <ImageIcon className="h-16 w-16 text-muted-foreground mb-4 mx-auto" />
                        )}
                        <motion.div
                          className="absolute -inset-4 bg-secondary/10 rounded-full blur-xl"
                          animate={{
                            scale: [1, 1.2, 1],
                            opacity: [0.5, 0.8, 0.5],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut",
                          }}
                        />
                      </div>
                    </motion.div>
                    <h2 className="text-2xl font-bold mb-2 text-center rowan-gradient-text">
                      {activeTab === "image-to-video" ? "Image to Video" : "Image to Image Edit"}
                    </h2>
                    <p className="text-muted-foreground text-center font-medium">
                      This feature is coming soon! 🚀
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Right Panel - Creative Guidance (280px fixed on desktop, collapsible on mobile) */}
          <AnimatePresence>
            {(showRightPanel || !isMobile) && (
              <motion.div
                className={`${
                  showRightPanel || !isMobile ? "block" : "hidden"
                } w-full lg:w-[280px] bg-card/50 backdrop-blur-sm border-l border-border/50 overflow-y-auto max-h-[50vh] lg:max-h-none`}
                variants={itemVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
              >
            <div className="p-4 sm:p-6 space-y-4 sm:space-y-6">
              {/* Pro Tips */}
              {showTips && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-muted/30 rounded-lg p-3 sm:p-4"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 bg-primary/20 rounded-lg">
                        <Lightbulb className="h-4 w-4 text-primary" />
                      </div>
                      <h3 className="font-semibold text-xs sm:text-sm text-primary">Pro Tips</h3>
                    </div>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowTips(false)}
                        className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                      >
                        ×
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowRightPanel(false)}
                        className="lg:hidden h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                      >
                        ×
                      </Button>
                    </div>
                  </div>
                  <ul className="space-y-2 text-[10px] sm:text-xs text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <Star className="h-3 w-3 text-primary mt-0.5 flex-shrink-0" />
                      <span>Be specific and descriptive in your prompts</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Star className="h-3 w-3 text-primary mt-0.5 flex-shrink-0" />
                      <span>Use negative prompts to exclude unwanted elements</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Star className="h-3 w-3 text-primary mt-0.5 flex-shrink-0" />
                      <span>Higher steps = better quality but slower generation</span>
                    </li>
                  </ul>
                </motion.div>
              )}

              {/* Example Prompts */}
              <motion.div variants={cardVariants}>
                <div className="bg-muted/30 rounded-lg p-3 sm:p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="p-1.5 bg-primary/20 rounded-lg">
                      <Info className="h-4 w-4 text-primary" />
                    </div>
                    <h3 className="font-semibold text-xs sm:text-sm">Example Prompts</h3>
                  </div>
                  <div className="space-y-1.5 sm:space-y-2">
                    {examplePrompts.map((example, index) => (
                      <motion.div
                        key={index}
                        whileHover={{ scale: !isMobile ? 1.02 : 1, x: !isMobile ? 3 : 0 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start text-left h-auto py-1.5 sm:py-2 px-2 sm:px-3 text-[10px] sm:text-xs hover:bg-primary/10 hover:text-primary border border-transparent hover:border-primary/20 transition-all rounded-lg"
                          onClick={() => {
                            setPrompt(example)
                            setShowRightPanel(false)
                          }}
                        >
                          <span className="truncate">{example}</span>
                        </Button>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Negative Prompt */}
              <motion.div className="space-y-2">
                <motion.div
                  whileHover={{ scale: !isMobile ? 1.02 : 1 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowNegativePrompt(!showNegativePrompt)}
                    className="w-full justify-between rounded-lg"
                  >
                    <span className="text-xs sm:text-sm">Negative Prompt</span>
                    <motion.div
                      animate={{ rotate: showNegativePrompt ? 180 : 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      {showNegativePrompt ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </motion.div>
                  </Button>
                </motion.div>
                <AnimatePresence>
                  {showNegativePrompt && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Textarea
                        placeholder="What to avoid in the image..."
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value.slice(0, 2000))}
                        className="min-h-[60px] sm:min-h-[80px] resize-none text-[10px] sm:text-xs rounded-lg"
                        maxLength={2000}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </div>
          </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Image Dialog with animations */}
      <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Image Details</DialogTitle>
            <DialogDescription>
              Seed: {selectedImage?.seed} • {selectedImage?.width}×{selectedImage?.height} • {selectedImage?.steps} steps
            </DialogDescription>
          </DialogHeader>
          <AnimatePresence mode="wait">
            {selectedImage && (
              <motion.div
                key={selectedImage.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="space-y-4"
              >
                <motion.div
                  className="relative w-full bg-muted rounded-lg overflow-hidden"
                  initial={{ scale: 0.95 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <img
                    src={selectedImage.imageBase64}
                    alt={selectedImage.prompt}
                    className="w-full h-auto"
                  />
                </motion.div>
                <div className="space-y-2">
                  <div>
                    <Label className="text-xs text-muted-foreground">Prompt</Label>
                    <p className="text-sm">{selectedImage.prompt}</p>
                  </div>
                  {selectedImage.negativePrompt && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Negative Prompt</Label>
                      <p className="text-sm">{selectedImage.negativePrompt}</p>
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      variant="outline"
                      onClick={() => {
                        navigator.clipboard.writeText(selectedImage.seed.toString())
                        toast.success("Seed copied")
                      }}
                    >
                      <Copy className="h-4 w-4 mr-2" />
                      Copy Seed
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      variant="outline"
                      onClick={() => handleDownload(selectedImage)}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </DialogContent>
      </Dialog>

      {/* Footer */}
      <Footer />
    </div>
  )
}
