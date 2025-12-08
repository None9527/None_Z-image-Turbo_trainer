import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

/**
 * GPU 信息接口
 */
export interface GpuInfo {
  name: string
  memoryTotal: number
  memoryUsed: number
  memoryPercent: number
  temperature: number
  utilization: number
}

/**
 * 系统信息接口
 */
export interface SystemInfo {
  python: string
  pytorch: string
  cuda: string
  cudnn: string
  platform: string
  diffusers: string
  transformers: string
  accelerate: string
  xformers: string
  bitsandbytes: string
}

/**
 * 模型状态接口
 */
export interface ModelStatus {
  downloaded: boolean
  path: string
  size_gb: number
}

/**
 * 下载状态接口
 */
export interface DownloadStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  progress: number
  downloaded_size_gb: number
  total_size_gb?: number
  speed?: number
  speed_unit?: string
  currentFile?: string
}

/**
 * 系统状态 Store
 * 纯数据存储，所有数据由 WebSocket 推送更新
 */
export const useSystemStore = defineStore('system', () => {
  // GPU 信息
  const gpuInfo = ref<GpuInfo>({
    name: '',
    memoryTotal: 0,
    memoryUsed: 0,
    memoryPercent: 0,
    temperature: 0,
    utilization: 0
  })

  // 系统信息
  const systemInfo = ref<SystemInfo>({
    python: '',
    pytorch: '',
    cuda: '',
    cudnn: '',
    platform: '',
    diffusers: '',
    transformers: '',
    accelerate: '',
    xformers: '',
    bitsandbytes: ''
  })

  // 模型状态
  const modelStatus = ref<ModelStatus>({
    downloaded: false,
    path: '',
    size_gb: 0
  })

  // 下载状态
  const downloadStatus = ref<DownloadStatus>({
    status: 'idle',
    progress: 0,
    downloaded_size_gb: 0
  })

  // 连接状态
  const isConnected = ref(false)

  // 计算属性
  const gpuMemoryText = computed(() => {
    return `${gpuInfo.value.memoryUsed.toFixed(1)} / ${gpuInfo.value.memoryTotal.toFixed(1)} GB`
  })

  const isDownloading = computed(() => downloadStatus.value.status === 'running')
  const isModelReady = computed(() => modelStatus.value.downloaded)

  // 更新方法 - 由 WebSocket store 调用
  function updateGpuInfo(info: Partial<GpuInfo>) {
    Object.assign(gpuInfo.value, info)
  }

  function updateSystemInfo(info: Partial<SystemInfo>) {
    Object.assign(systemInfo.value, info)
  }

  function updateModelStatus(status: Partial<ModelStatus>) {
    Object.assign(modelStatus.value, status)
  }

  function updateDownloadStatus(status: Partial<DownloadStatus>) {
    Object.assign(downloadStatus.value, status)
  }

  function setConnected(connected: boolean) {
    isConnected.value = connected
  }

  return {
    // 状态
    gpuInfo,
    systemInfo,
    modelStatus,
    downloadStatus,
    isConnected,
    
    // 计算属性
    gpuMemoryText,
    isDownloading,
    isModelReady,
    
    // 更新方法
    updateGpuInfo,
    updateSystemInfo,
    updateModelStatus,
    updateDownloadStatus,
    setConnected
  }
})
