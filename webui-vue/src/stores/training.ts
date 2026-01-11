import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export interface TrainingConfig {
  // 基础配置
  outputDir: string
  outputName: string

  // 模型配置
  modelType: 'zimage'  // 模型类型 (仅支持 Z-Image)
  modelPath: string
  vaePath: string
  textEncoderPath: string

  // 数据集配置
  datasetConfigPath: string
  cacheDir: string

  // 训练参数
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: string
  scheduler: string
  warmupSteps: number

  // LoRA 配置
  networkDim: number
  networkAlpha: number

  // 高级配置
  mixedPrecision: 'fp16' | 'bf16' | 'fp32'
  gradientCheckpointing: boolean
  gradientAccumulationSteps: number
  maxGradNorm: number
  seed: number
}

export interface TrainingProgress {
  isRunning: boolean
  isLoading: boolean  // 模型加载中
  currentEpoch: number
  totalEpochs: number
  currentStep: number
  totalSteps: number
  loss: number
  learningRate: number
  elapsedTime: number
  estimatedTimeRemaining: number
  lossHistory: number[]
  lrHistory: number[]
}

export const useTrainingStore = defineStore('training', () => {
  const config = ref<TrainingConfig>({
    outputDir: './output',
    outputName: 'zimage-lora',
    modelType: 'zimage',
    modelPath: '',
    vaePath: '',
    textEncoderPath: '',
    datasetConfigPath: './dataset_config.toml',
    cacheDir: './cache',
    epochs: 10,
    batchSize: 1,
    learningRate: 1e-4,
    optimizer: 'adamw',
    scheduler: 'cosine',
    warmupSteps: 100,
    networkDim: 64,
    networkAlpha: 64,
    mixedPrecision: 'bf16',
    gradientCheckpointing: true,
    gradientAccumulationSteps: 1,
    maxGradNorm: 1.0,
    seed: 42
  })

  const progress = ref<TrainingProgress>({
    isRunning: false,
    isLoading: false,
    currentEpoch: 0,
    totalEpochs: 0,
    currentStep: 0,
    totalSteps: 0,
    loss: 0,
    learningRate: 0,
    elapsedTime: 0,
    estimatedTimeRemaining: 0,
    lossHistory: [],
    lrHistory: []
  })

  const progressPercent = computed(() => {
    if (progress.value.totalSteps === 0) return 0
    return Math.round((progress.value.currentStep / progress.value.totalSteps) * 100)
  })

  async function loadConfig(path: string) {
    try {
      const response = await axios.get(`/api/config/load?path=${encodeURIComponent(path)}`)
      config.value = { ...config.value, ...response.data }
      return true
    } catch (error) {
      console.error('Failed to load config:', error)
      return false
    }
  }

  async function saveConfig(path: string) {
    try {
      await axios.post('/api/config/save', { path, config: config.value })
      return true
    } catch (error) {
      console.error('Failed to save config:', error)
      return false
    }
  }

  async function startTraining() {
    try {
      const response = await axios.post('/api/training/start', config.value)
      progress.value.isRunning = true
      return response.data
    } catch (error) {
      console.error('Failed to start training:', error)
      throw error
    }
  }

  async function stopTraining() {
    try {
      await axios.post('/api/training/stop')
      progress.value.isRunning = false
    } catch (error) {
      console.error('Failed to stop training:', error)
      throw error
    }
  }

  function updateProgress(data: Partial<TrainingProgress>) {
    // 只更新基础字段，不 push 历史数据
    // 历史数据由后端 training_history 维护，通过 setHistory 设置
    progress.value = { ...progress.value, ...data }
  }

  function setHistory(lossHistory: number[], lrHistory: number[]) {
    // 从后端恢复完整的历史数据
    progress.value.lossHistory = lossHistory
    progress.value.lrHistory = lrHistory
  }

  function appendHistory(loss?: number, lr?: number) {
    // 追加单个历史数据点（用于实时更新）
    if (loss !== undefined) {
      progress.value.lossHistory.push(loss)
    }
    if (lr !== undefined) {
      progress.value.lrHistory.push(lr)
    }
  }

  async function fetchDefaults() {
    try {
      const response = await axios.get('/api/training/defaults')
      config.value = { ...config.value, ...response.data }
    } catch (error) {
      console.error('Failed to fetch defaults:', error)
    }
  }

  // Initialize with defaults
  fetchDefaults()

  return {
    config,
    progress,
    progressPercent,
    loadConfig,
    saveConfig,
    startTraining,
    stopTraining,
    updateProgress,
    setHistory,
    appendHistory,
    fetchDefaults
  }
})

