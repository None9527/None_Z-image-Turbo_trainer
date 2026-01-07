<template>
  <div class="generation-container">
    <div class="main-content">
      <div class="generation-grid">
        <!-- Controls Section -->
        <div class="controls-section">
          <el-card class="params-card glass-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span><el-icon><Operation /></el-icon> 生成参数</span>
              </div>
            </template>
            
            <el-form :model="params" size="small" class="params-form">
              <!-- Model Type Selector -->
              <div class="param-group">
                <div class="group-label">模型类型 (MODEL)</div>
                <el-radio-group v-model="params.model_type" class="model-selector">
                  <el-radio-button label="zimage">
                    <el-icon><Aim /></el-icon> Z-Image
                  </el-radio-button>
                  <el-radio-button label="longcat">
                    <el-icon><MagicStick /></el-icon> LongCat
                  </el-radio-button>
                </el-radio-group>
              </div>

              <!-- Prompt -->
              <div class="param-group">
                <div class="group-label">提示词 (PROMPT)</div>
                <el-input
                  v-model="params.prompt"
                  type="textarea"
                  :rows="4"
                  placeholder="描述你想要生成的图片..."
                  resize="none"
                  class="prompt-input"
                />
              </div>

              <!-- LoRA -->
              <div class="param-group">
                <div class="group-label">LORA 模型</div>
                <el-select 
                  v-model="params.lora_path" 
                  placeholder="选择 LoRA 模型..." 
                  clearable 
                  filterable
                  style="width: 100%; margin-bottom: 8px;"
                >
                  <el-option
                    v-for="lora in loraList"
                    :key="lora.path"
                    :label="lora.name"
                    :value="lora.path"
                  >
                    <span style="float: left">{{ lora.name }}</span>
                    <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">
                      {{ (lora.size / 1024 / 1024).toFixed(1) }} MB
                    </span>
                  </el-option>
                </el-select>

                <div v-if="params.lora_path" class="lora-settings">
                  <div class="control-row">
                    <span class="label">权重</span>
                    <el-slider v-model="params.lora_scale" :min="0" :max="2" :step="0.05" :show-tooltip="false" class="slider-flex" />
                    <el-input-number v-model="params.lora_scale" :min="0" :max="2" :step="0.05" controls-position="right" class="input-fixed" />
                  </div>
                  <div class="control-row" style="margin-top: 8px;">
                    <span class="label">对比</span>
                    <el-switch v-model="params.comparison_mode" active-text="生成原图对比" />
                  </div>
                </div>
              </div>

              <!-- Resolution -->
              <div class="param-group">
                <div class="group-label">分辨率 (RESOLUTION)</div>
                
                <!-- Aspect Ratio Grid -->
                <div class="ratio-grid">
                  <el-button 
                    v-for="ratio in aspectRatios" 
                    :key="ratio.label"
                    size="small"
                    :type="currentRatio === ratio.label ? 'primary' : 'default'"
                    @click="setAspectRatio(ratio)"
                    class="ratio-btn"
                  >
                    {{ ratio.label }}
                  </el-button>
                </div>

                <!-- Width Row -->
                <div class="control-row">
                  <span class="label">宽度</span>
                  <el-slider v-model="params.width" :min="256" :max="2048" :step="64" :marks="resolutionMarks" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.width" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
                </div>

                <!-- Height Row -->
                <div class="control-row">
                  <span class="label">高度</span>
                  <el-slider v-model="params.height" :min="256" :max="2048" :step="64" :marks="resolutionMarks" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.height" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
                </div>
              </div>

              <!-- Settings -->
              <div class="param-group">
                <div class="group-label">生成设置 (SETTINGS)</div>
                
                <!-- Steps Row -->
                <div class="control-row">
                  <span class="label">步数</span>
                  <el-slider v-model="params.steps" :min="1" :max="50" :step="1" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.steps" :min="1" :max="50" controls-position="right" class="input-fixed" />
                </div>

                <!-- CFG Row -->
                <div class="control-row">
                  <span class="label">引导</span>
                  <el-slider v-model="params.guidance_scale" :min="1" :max="20" :step="0.5" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.guidance_scale" :min="1" :max="20" :step="0.5" controls-position="right" class="input-fixed" />
                </div>

                <!-- Seed Row -->
                <div class="control-row">
                  <span class="label">Seed</span>
                  <div class="seed-wrapper">
                    <el-input-number v-model="params.seed" :min="-1" placeholder="随机" controls-position="right" class="seed-input" />
                    <el-button @click="params.seed = -1" icon="Refresh" size="small" class="seed-btn" />
                  </div>
                </div>
              </div>
              
              <el-button 
                type="primary" 
                size="large" 
                class="generate-btn" 
                @click="generateImage" 
                :loading="generating"
                :disabled="generating"
              >
                <el-icon><MagicStick /></el-icon>
                {{ generating ? '生成中...' : '立即生成' }}
              </el-button>
            </el-form>
          </el-card>
        </div>

        <!-- Preview Section -->
        <div class="preview-section">
          <el-card class="preview-card glass-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span>生成结果</span>
                <div class="header-actions">
                  <span v-if="resultImage" class="res-tag">{{ params.width }} x {{ params.height }}</span>
                  <el-button v-if="resultImage" type="success" size="small" @click="downloadImage(resultImage)">
                    <el-icon><Download /></el-icon> 下载
                  </el-button>
                </div>
              </div>
            </template>
            
            <div 
              class="image-container" 
              @wheel.prevent="handleWheel($event, 'main')"
              @mousedown="startDrag($event, 'main')"
              @mousemove="onDrag($event, 'main')"
              @mouseup="stopDrag('main')"
              @mouseleave="stopDrag('main')"
              @dblclick="resetZoom('main')"
            >
              <div class="zoom-wrapper" :style="mainImageStyle">
                <!-- 对比模式：双图展示 -->
                <div v-if="isComparisonResult && comparisonImages.length === 2" class="comparison-container">
                  <div class="comparison-image-wrapper">
                    <div class="comparison-label original-label">
                      <el-icon><Picture /></el-icon>
                      原始模型 (无 LoRA)
                    </div>
                    <img :src="comparisonImages[0].image" class="comparison-image" draggable="false" />
                  </div>
                  <div class="comparison-divider"></div>
                  <div class="comparison-image-wrapper">
                    <div class="comparison-label lora-label">
                      <el-icon><MagicStick /></el-icon>
                      LoRA: {{ getLoraFileName(comparisonImages[1].lora_path) }}
                      <span class="lora-weight">(权重: {{ comparisonImages[1].lora_scale?.toFixed(2) || params.lora_scale.toFixed(2) }})</span>
                    </div>
                    <img :src="comparisonImages[1].image" class="comparison-image" draggable="false" />
                  </div>
                </div>
                <!-- 普通模式：单图展示 -->
                <img v-else-if="resultImage" :src="resultImage" class="generated-image" alt="Generated Image" draggable="false" />
                <div v-else class="placeholder">
                  <el-icon class="placeholder-icon"><Picture /></el-icon>
                  <p>生成的图片将显示在这里</p>
                </div>
              </div>
              
              <!-- 生成中覆盖层 -->
              <Transition name="fade">
                <div v-if="generating" class="generation-overlay">
                  <div class="generation-progress-card">
                    <div class="progress-icon">
                      <el-icon class="spinning"><Loading /></el-icon>
                    </div>
                    <div class="progress-info">
                      <div class="progress-stage">{{ progressStage }}</div>
                      <div class="progress-detail">{{ progressMessage }}</div>
                      <el-progress 
                        v-if="progressTotal > 0"
                        :percentage="Math.round((progressStep / progressTotal) * 100)"
                        :stroke-width="8"
                        :show-text="true"
                        style="margin-top: 12px; width: 200px;"
                      />
                    </div>
                  </div>
                </div>
              </Transition>
              
              <!-- Zoom Controls Overlay -->
              <div class="zoom-controls" v-if="resultImage && !generating">
                <el-button-group>
                  <el-button size="small" @click="zoomOut('main')"><el-icon><Minus /></el-icon></el-button>
                  <el-button size="small" @click="resetZoom('main')">{{ Math.round(mainScale * 100) }}%</el-button>
                  <el-button size="small" @click="zoomIn('main')"><el-icon><Plus /></el-icon></el-button>
                </el-button-group>
              </div>
            </div>
            
            <div class="result-info" v-if="resultSeed !== null">
              <span>Seed: {{ resultSeed }}</span>
            </div>
          </el-card>
        </div>
      </div>

      <!-- History Section -->
      <div class="history-section">
        <div class="section-header">
          <h3><el-icon><Clock /></el-icon> 历史记录</h3>
          <el-button link @click="fetchHistory" :loading="loadingHistory">
            <el-icon><Refresh /></el-icon> 刷新
          </el-button>
        </div>
        
        <div class="history-grid" v-loading="loadingHistory">
          <!-- 进行中/已中断的任务 -->
          <div v-if="pendingTask" class="history-card glass-card pending-task-card" :class="{ 'interrupted': pendingTask.interrupted }">
            <div class="history-thumb-wrapper pending-thumb">
              <div class="pending-overlay">
                <template v-if="!pendingTask.interrupted">
                  <el-icon class="spinning pending-icon"><Loading /></el-icon>
                  <span class="pending-text">生成中...</span>
                </template>
                <template v-else>
                  <el-icon class="pending-icon interrupted-icon"><WarningFilled /></el-icon>
                  <span class="pending-text">已中断</span>
                </template>
              </div>
            </div>
            <div class="history-info">
              <div class="history-prompt" :title="pendingTask.prompt">{{ pendingTask.prompt }}</div>
              <div class="history-meta">
                <span>{{ pendingTask.width }}x{{ pendingTask.height }}</span>
                <el-button v-if="pendingTask.interrupted" type="primary" size="small" @click.stop="retryPendingTask">
                  重新生成
                </el-button>
                <el-button v-if="pendingTask.interrupted" type="info" size="small" @click.stop="clearPendingTask">
                  取消
                </el-button>
              </div>
            </div>
          </div>
          
          <div v-if="historyList.length === 0 && !pendingTask" class="empty-history">
            暂无历史记录
          </div>
          <div 
            v-for="item in historyList" 
            :key="item.url" 
            class="history-card glass-card"
            @click="openLightbox(item)"
          >
            <div class="history-thumb-wrapper">
              <el-image 
                :src="`${item.thumbnail}`" 
                fit="cover" 
                class="history-thumb"
                lazy
              />
              <div class="history-overlay">
                <el-icon class="overlay-icon"><ZoomIn /></el-icon>
              </div>
            </div>
            <div class="history-info">
              <div class="history-prompt" :title="item.metadata.prompt">{{ item.metadata.prompt }}</div>
              <div class="history-meta">
                <el-tag size="small" :type="(item.metadata.model_type === 'longcat' ? 'warning' : 'primary') as 'warning' | 'primary'" effect="plain">
                  {{ item.metadata.model_type === 'longcat' ? 'LongCat' : 'Z-Image' }}
                </el-tag>
                <span>{{ item.metadata.width }}x{{ item.metadata.height }}</span>
                <span>Seed: {{ item.metadata.seed }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Lightbox Overlay -->
    <div v-if="lightboxVisible" class="lightbox-overlay" @click.self="closeLightbox">
      <div class="lightbox-content">
        <div class="lightbox-header">
          <div class="lightbox-title">历史记录详情</div>
          <div class="lightbox-actions">
            <el-button type="primary" @click="restoreParams(lightboxItem.metadata)">
              应用此参数
            </el-button>
            <el-button type="success" @click="downloadImage(`${lightboxItem.url}`)">
              <el-icon><Download /></el-icon> 下载
            </el-button>
             <el-button type="danger" @click="deleteHistoryItem(lightboxItem, true)">
              <el-icon><Delete /></el-icon> 删除
            </el-button>
            <el-button circle @click="closeLightbox">
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
        </div>
        
        <div 
          class="lightbox-image-container"
          @wheel.prevent="handleWheel($event, 'lightbox')"
          @mousedown="startDrag($event, 'lightbox')"
          @mousemove="onDrag($event, 'lightbox')"
          @mouseup="stopDrag('lightbox')"
          @mouseleave="stopDrag('lightbox')"
          @dblclick="resetZoom('lightbox')"
        >
          <div class="zoom-wrapper" :style="lightboxImageStyle">
            <img :src="`${lightboxItem.url}`" class="lightbox-image" draggable="false" />
          </div>
          
           <!-- Zoom Controls Overlay -->
           <div class="zoom-controls">
            <el-button-group>
              <el-button size="small" @click="zoomOut('lightbox')"><el-icon><Minus /></el-icon></el-button>
              <el-button size="small" @click="resetZoom('lightbox')">{{ Math.round(lightboxScale * 100) }}%</el-button>
              <el-button size="small" @click="zoomIn('lightbox')"><el-icon><Plus /></el-icon></el-button>
            </el-button-group>
          </div>
        </div>
        
        <div class="lightbox-info">
          <div class="info-item">
            <span class="label">Prompt:</span>
            <span class="text">{{ lightboxItem.metadata.prompt }}</span>
          </div>
          <div class="info-row">
            <div class="info-item">
              <span class="label">Size:</span>
              <span class="text">{{ lightboxItem.metadata.width }} x {{ lightboxItem.metadata.height }}</span>
            </div>
            <div class="info-item">
              <span class="label">Steps:</span>
              <span class="text">{{ lightboxItem.metadata.steps }}</span>
            </div>
            <div class="info-item">
              <span class="label">CFG:</span>
              <span class="text">{{ lightboxItem.metadata.guidance_scale }}</span>
            </div>
            <div class="info-item">
              <span class="label">Seed:</span>
              <span class="text">{{ lightboxItem.metadata.seed }}</span>
            </div>
            <div class="info-item">
              <span class="label">Model:</span>
              <el-tag size="small" :type="(lightboxItem.metadata.model_type === 'longcat' ? 'warning' : 'primary') as 'warning' | 'primary'">
                {{ lightboxItem.metadata.model_type === 'longcat' ? 'LongCat' : 'Z-Image' }}
              </el-tag>
            </div>
            <div class="info-item" v-if="lightboxItem.metadata.lora_path">
              <span class="label">LoRA:</span>
              <span class="text">{{ lightboxItem.metadata.lora_path.split('/').pop() }} ({{ lightboxItem.metadata.lora_scale }})</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, reactive, watch } from 'vue'
import { MagicStick, Download, Picture, Refresh, Clock, Plus, Minus, ZoomIn, Close, Delete, Operation, Loading, WarningFilled, Aim } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useWebSocketStore } from '@/stores/websocket'

const wsStore = useWebSocketStore()

// localStorage keys
const STORAGE_KEY_PARAMS = 'generation_params'
const STORAGE_KEY_RESULT = 'generation_result'
const STORAGE_KEY_PENDING = 'generation_pending'

// 从 localStorage 加载保存的参数
const loadSavedParams = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_PARAMS)
    if (saved) {
      return JSON.parse(saved)
    }
  } catch (e) {
    console.warn('Failed to load saved params:', e)
  }
  return null
}

// 从 localStorage 加载保存的结果
const loadSavedResult = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_RESULT)
    if (saved) {
      return JSON.parse(saved)
    }
  } catch (e) {
    console.warn('Failed to load saved result:', e)
  }
  return null
}

const defaultParams = {
  prompt: "A futuristic city with flying cars, cyberpunk style, highly detailed, 8k",
  negative_prompt: "",
  steps: 9,
  guidance_scale: 1.0,
  seed: -1,
  width: 1024,
  height: 1024,
  lora_path: null as string | null,
  lora_scale: 1.0,
  comparison_mode: false,
  model_type: "zimage" as "zimage" | "longcat"
}

const savedParams = loadSavedParams()
const params = ref(savedParams ? { ...defaultParams, ...savedParams } : { ...defaultParams })

const generating = ref(false)
const savedResult = loadSavedResult()
const resultImage = ref<string | null>(savedResult?.image || null)
const resultSeed = ref<number | null>(savedResult?.seed || null)

// 对比模式结果
const comparisonImages = ref<{image: string, lora_path: string | null, lora_scale?: number}[]>([])
const isComparisonResult = ref(false)

// 获取 LoRA 文件名
const getLoraFileName = (path: string | null) => {
  if (!path) return 'Unknown'
  const parts = path.split(/[\/\\]/)
  const filename = parts[parts.length - 1]
  return filename.replace('.safetensors', '')
}

// SSE 进度状态
const progressStage = ref('准备中...')
const progressMessage = ref('')
const progressStep = ref(0)
const progressTotal = ref(0)

// 监听参数变化，自动保存到 localStorage
watch(params, (newParams) => {
  try {
    localStorage.setItem(STORAGE_KEY_PARAMS, JSON.stringify(newParams))
  } catch (e) {
    console.warn('Failed to save params:', e)
  }
}, { deep: true })

// Pending task state
interface PendingTask {
  prompt: string
  width: number
  height: number
  steps: number
  seed: number
  startTime: number
  interrupted: boolean
}

const pendingTask = ref<PendingTask | null>(null)

// 检查是否有被中断的任务
const checkPendingTask = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_PENDING)
    if (saved) {
      const task = JSON.parse(saved)
      // 如果页面刚加载，说明之前的任务被中断了
      task.interrupted = true
      pendingTask.value = task
    }
  } catch (e) {
    console.warn('Failed to load pending task:', e)
  }
}

// 保存 pending task
const savePendingTask = () => {
  const task: PendingTask = {
    prompt: params.value.prompt,
    width: params.value.width,
    height: params.value.height,
    steps: params.value.steps,
    seed: params.value.seed,
    startTime: Date.now(),
    interrupted: false
  }
  pendingTask.value = task
  try {
    localStorage.setItem(STORAGE_KEY_PENDING, JSON.stringify(task))
  } catch (e) {
    console.warn('Failed to save pending task:', e)
  }
}

// 清除 pending task
const clearPendingTask = () => {
  pendingTask.value = null
  try {
    localStorage.removeItem(STORAGE_KEY_PENDING)
  } catch (e) {
    console.warn('Failed to clear pending task:', e)
  }
}

// 重试被中断的任务
const retryPendingTask = () => {
  if (pendingTask.value) {
    // 恢复参数
    params.value.prompt = pendingTask.value.prompt
    params.value.width = pendingTask.value.width
    params.value.height = pendingTask.value.height
    params.value.steps = pendingTask.value.steps
    params.value.seed = pendingTask.value.seed
    clearPendingTask()
    // 重新生成
    generateImage()
  }
}


// History state
const loadingHistory = ref(false)
const historyList = ref<any[]>([])

// LoRA state
const loraList = ref<any[]>([])

// Lightbox state
const lightboxVisible = ref(false)
const lightboxItem = ref<any>(null)

// Zoom state management
const createZoomState = () => reactive({
  scale: 1,
  translateX: 0,
  translateY: 0,
  isDragging: false,
  startX: 0,
  startY: 0
})

const mainZoom = createZoomState()
const lightboxZoom = createZoomState()

const mainImageStyle = computed(() => ({
  transform: `translate(${mainZoom.translateX}px, ${mainZoom.translateY}px) scale(${mainZoom.scale})`,
  transition: mainZoom.isDragging ? 'none' : 'transform 0.1s ease-out'
}))

const lightboxImageStyle = computed(() => ({
  transform: `translate(${lightboxZoom.translateX}px, ${lightboxZoom.translateY}px) scale(${lightboxZoom.scale})`,
  transition: lightboxZoom.isDragging ? 'none' : 'transform 0.1s ease-out'
}))

// Aspect Ratio Logic
const aspectRatios = [
  { label: '1:1', w: 1024, h: 1024 },
  { label: '4:3', w: 1024, h: 768 },
  { label: '3:4', w: 768, h: 1024 },
  { label: '16:9', w: 1024, h: 576 },
  { label: '9:16', w: 576, h: 1024 }
]

const currentRatio = computed(() => {
  const { width, height } = params.value
  const match = aspectRatios.find(r => r.w === width && r.h === height)
  return match ? match.label : 'Custom'
})

const resolutionMarks = {
  512: '',
  1024: '',
  1536: '',
  2048: ''
}

const setAspectRatio = (ratio: any) => {
  params.value.width = ratio.w
  params.value.height = ratio.h
}


// Generation Logic - 使用 SSE 流式接口获取实时进度
const generateImage = async () => {
  if (!params.value.prompt) {
    ElMessage.warning('请输入提示词')
    return
  }
  
  generating.value = true
  isComparisonResult.value = false
  comparisonImages.value = []
  progressStage.value = '准备中...'
  progressMessage.value = ''
  progressStep.value = 0
  progressTotal.value = params.value.steps
  savePendingTask()
  
  try {
    // 构建查询参数
    const queryParams = new URLSearchParams()
    Object.entries(params.value).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        queryParams.append(key, String(value))
      }
    })
    
    // 使用 fetch 进行 SSE 流式请求
    const response = await fetch('/api/generate-stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params.value),
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    
    if (!reader) {
      throw new Error('Failed to get reader')
    }
    
    // 使用缓冲区处理可能被分割的 SSE 消息
    let buffer = ''
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      
      // 按双换行符分割完整的 SSE 消息
      const messages = buffer.split('\n\n')
      // 最后一个可能是不完整的消息，保留在缓冲区
      buffer = messages.pop() || ''
      
      for (const message of messages) {
        const lines = message.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // 处理进度更新
              if (data.stage) {
                switch (data.stage) {
                  case 'loading':
                    progressStage.value = '🔄 加载模型...'
                    break
                  case 'generating':
                    progressStage.value = '🎨 生成中...'
                    progressStep.value = data.step || 0
                    progressTotal.value = data.total || params.value.steps
                    break
                  case 'saving':
                    progressStage.value = '💾 保存中...'
                    break
                  case 'completed':
                    progressStage.value = '✅ 完成!'
                    break
                  case 'error':
                    progressStage.value = '❌ 错误'
                    break
                }
                progressMessage.value = data.message || ''
              }
              
              // 处理最终结果
              if (data.success !== undefined) {
                if (data.success) {
                  if (data.comparison_mode && data.images) {
                    // 对比模式：两张图
                    isComparisonResult.value = true
                    comparisonImages.value = data.images.map((img: any) => ({
                      image: img.image.startsWith('data:') ? img.image : `data:image/png;base64,${img.image}`,
                      lora_path: img.lora_path,
                      lora_scale: img.lora_scale || params.value.lora_scale,
                    }))
                    // 显示第二张图（LoRA）作为主结果
                    if (data.images.length > 1) {
                      resultImage.value = comparisonImages.value[1].image
                    } else {
                      resultImage.value = comparisonImages.value[0].image
                    }
                    resultSeed.value = data.seed
                  } else {
                    // 普通模式
                    isComparisonResult.value = false
                    const imgData = data.image || ''
                    resultImage.value = imgData.startsWith('data:') ? imgData : `data:image/png;base64,${imgData}`
                    resultSeed.value = data.seed
                  }
                  
                  // 保存结果
                  try {
                    localStorage.setItem(STORAGE_KEY_RESULT, JSON.stringify({
                      image: resultImage.value,
                      seed: resultSeed.value
                    }))
                  } catch (e) {
                    console.warn('Failed to save result:', e)
                  }
                  
                  ElMessage.success('生成成功！')
                  fetchHistory()
                  resetZoom('main')
                } else {
                  ElMessage.error('生成失败: ' + (data.message || data.error || 'Unknown error'))
                }
              }
            } catch (e) {
              // JSON 解析错误，忽略（可能是不完整的数据）
              console.debug('SSE parse error (may be incomplete):', e)
            }
          }
        }
      }
    }
  } catch (e: any) {
    console.error('Generation error:', e)
    ElMessage.error('生成失败: ' + e.message)
  } finally {
    generating.value = false
    clearPendingTask()
  }
}

const downloadImage = (url: string | null) => {
  if (!url) return
  
  const link = document.createElement('a')
  link.href = url
  link.download = `generated_${Date.now()}.png`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

// History Logic
const fetchHistory = async () => {
  loadingHistory.value = true
  try {
    const res = await axios.get('/api/history')
    historyList.value = res.data.history || res.data || []
  } catch (e) {
    console.error('Failed to fetch history:', e)
    ElMessage.error('获取历史记录失败')
  } finally {
    loadingHistory.value = false
  }
}

const deleteHistoryItem = async (item: any, fromLightbox = false) => {
  try {
    await ElMessageBox.confirm(
      '确定要删除这张图片吗？此操作不可恢复。',
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger'
      }
    )
    
    const res = await axios.post('/api/history/delete', {
      timestamps: [item.metadata.timestamp]
    })
    
    if (res.data.success) {
      ElMessage.success('删除成功')
      if (fromLightbox) {
        closeLightbox()
      }
      fetchHistory()
    } else {
      ElMessage.error('删除失败')
    }
  } catch (e) {
    if (e !== 'cancel') {
      console.error('Delete error:', e)
      ElMessage.error('删除请求失败')
    }
  }
}

// LoRA Logic
const fetchLoras = async () => {
  try {
    const res = await axios.get('/api/loras')
    // 新的返回格式: { loras, loraPath, loraPathExists }
    loraList.value = res.data.loras || res.data || []
  } catch (e) {
    console.error('Failed to fetch LoRAs:', e)
  }
}

const openLightbox = (item: any) => {
  lightboxItem.value = item
  lightboxVisible.value = true
  resetZoom('lightbox')
  document.body.style.overflow = 'hidden'
}

const closeLightbox = () => {
  lightboxVisible.value = false
  lightboxItem.value = null
  document.body.style.overflow = ''
}

const restoreParams = (metadata: any) => {
  params.value.prompt = metadata.prompt
  params.value.steps = metadata.steps
  params.value.guidance_scale = metadata.guidance_scale
  params.value.seed = metadata.seed
  params.value.width = metadata.width
  params.value.height = metadata.height
  params.value.model_type = metadata.model_type || "zimage"
  if (metadata.lora_path) {
    params.value.lora_path = metadata.lora_path
    params.value.lora_scale = metadata.lora_scale || 1.0
    params.value.comparison_mode = metadata.comparison_mode || false
  } else {
    params.value.lora_path = null
    params.value.lora_scale = 1.0
    params.value.comparison_mode = false
  }
  
  ElMessage.success('参数已应用')
  closeLightbox()
}

// Zoom Handlers
const getZoomState = (target: 'main' | 'lightbox') => target === 'main' ? mainZoom : lightboxZoom
const getScale = (target: 'main' | 'lightbox') => target === 'main' ? mainScale : lightboxScale

const mainScale = computed(() => mainZoom.scale)
const lightboxScale = computed(() => lightboxZoom.scale)

const handleWheel = (e: WheelEvent, target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  const newScale = Math.min(Math.max(state.scale * delta, 0.5), 5)
  state.scale = newScale
}

const startDrag = (e: MouseEvent, target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  if (state.scale <= 1) return
  state.isDragging = true
  state.startX = e.clientX - state.translateX
  state.startY = e.clientY - state.translateY
}

const onDrag = (e: MouseEvent, target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  if (!state.isDragging) return
  state.translateX = e.clientX - state.startX
  state.translateY = e.clientY - state.startY
}

const stopDrag = (target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  state.isDragging = false
}

const resetZoom = (target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  state.scale = 1
  state.translateX = 0
  state.translateY = 0
}

const zoomIn = (target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  state.scale = Math.min(state.scale * 1.2, 5)
}

const zoomOut = (target: 'main' | 'lightbox') => {
  const state = getZoomState(target)
  state.scale = Math.max(state.scale * 0.8, 0.5)
}

onMounted(() => {
  fetchHistory()
  fetchLoras()
  checkPendingTask() // 检查是否有被中断的任务
})
</script>

<style scoped>
.generation-container {
  padding: 24px;
  height: 100%;
  overflow-y: auto;
}

.main-content {
  max-width: 1600px;
  margin: 0 auto;
}

.generation-grid {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 24px;
  margin-bottom: 40px;
  min-height: 600px;
}

.params-card {
  height: fit-content;
  position: sticky;
  top: 0;
}

.preview-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 600px;
}

.preview-card :deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
  background: #000;
  position: relative;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
  font-size: 15px;
}

.card-header span {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Professional Params Layout */
.params-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.param-group {
  background: var(--el-fill-color-lighter);
  padding: 16px;
  border-radius: 8px;
  border: 1px solid var(--el-border-color-lighter);
}

.group-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--el-text-color-secondary);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.prompt-input :deep(.el-textarea__inner) {
  background: var(--el-bg-color);
  border-color: var(--el-border-color-light);
  font-family: inherit;
}

.model-selector {
  width: 100%;
  display: flex;
}

.model-selector :deep(.el-radio-button) {
  flex: 1;
}

.model-selector :deep(.el-radio-button__inner) {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.ratio-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4px;
  margin-bottom: 16px;
}

.ratio-btn {
  width: 100%;
  padding: 6px 0;
  font-size: 11px;
}

.control-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.control-row:last-child {
  margin-bottom: 0;
}

.control-row .label {
  font-size: 12px;
  color: var(--el-text-color-regular);
  width: 32px;
  flex-shrink: 0;
}

.slider-flex {
  flex: 1;
  margin-right: 8px;
}

.input-fixed {
  width: 80px !important;
}

.seed-wrapper {
  flex: 1;
  display: flex;
  gap: 8px;
}

.seed-input {
  flex: 1;
}

.seed-btn {
  width: 32px;
  padding: 0;
}

.lora-settings {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--el-border-color-light);
}

.generate-btn {
  width: 100%;
  font-weight: bold;
  letter-spacing: 1px;
  height: 48px;
  font-size: 16px;
  margin-top: 8px;
  box-shadow: 0 4px 12px rgba(var(--el-color-primary-rgb), 0.3);
}

/* Image Preview & Zoom */
.image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  background-image: 
    linear-gradient(45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(-45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #1a1a1a 75%), 
    linear-gradient(-45deg, transparent 75%, #1a1a1a 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  background-color: #111;
  cursor: grab;
}

.image-container:active {
  cursor: grabbing;
}

.zoom-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.generated-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  box-shadow: 0 0 30px rgba(0,0,0,0.5);
  pointer-events: none;
}

/* 对比模式样式 */
.comparison-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 100%;
  width: 100%;
}

.comparison-image-wrapper {
  position: relative;
  flex: 1;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.comparison-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  box-shadow: 0 0 20px rgba(0,0,0,0.5);
}

.comparison-label {
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  z-index: 10;
  backdrop-filter: blur(4px);
}

.comparison-label.lora-label {
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.9), rgba(103, 194, 58, 0.9));
}

.comparison-divider {
  width: 2px;
  height: 80%;
  background: linear-gradient(to bottom, transparent, rgba(255,255,255,0.5), transparent);
  flex-shrink: 0;
}

.zoom-controls {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.6);
  border-radius: 4px;
  padding: 4px;
  backdrop-filter: blur(4px);
}

/* 生成进度覆盖层 */
.generation-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  backdrop-filter: blur(8px);
}

.generation-progress-card {
  background: linear-gradient(135deg, rgba(30, 30, 40, 0.95), rgba(20, 20, 30, 0.98));
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 32px 48px;
  min-width: 320px;
  max-width: 400px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(var(--el-color-primary-rgb), 0.15);
  text-align: center;
}

.progress-icon {
  font-size: 48px;
  color: var(--el-color-primary);
  margin-bottom: 20px;
}

.progress-icon .spinning {
  animation: spin 1.5s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.progress-info {
  margin-bottom: 24px;
}

.progress-stage {
  font-size: 20px;
  font-weight: bold;
  color: #fff;
  margin-bottom: 8px;
}

.progress-detail {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.7);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-bar-container .el-progress {
  flex: 1;
}

.progress-bar-container .progress-percent {
  font-size: 16px;
  font-weight: bold;
  color: var(--el-color-primary);
  font-family: 'SF Mono', 'Fira Code', monospace;
  min-width: 48px;
  text-align: right;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: var(--el-text-color-secondary);
  opacity: 0.5;
}

.placeholder-icon {
  font-size: 80px;
  margin-bottom: 16px;
}

.result-info {
  padding: 8px 16px;
  background: var(--el-bg-color-overlay);
  border-top: 1px solid var(--el-border-color-light);
  text-align: right;
  font-family: monospace;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  z-index: 10;
}

/* History Section */
.history-section {
  margin-top: 40px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h3 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
}

.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
}

.history-card {
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  overflow: hidden;
  border-radius: 8px;
  border: 1px solid var(--el-border-color-light);
  background: var(--el-bg-color);
}

.history-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.history-thumb-wrapper {
  position: relative;
  aspect-ratio: 1;
  overflow: hidden;
}

.history-thumb {
  width: 100%;
  height: 100%;
  display: block;
}

.history-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
  color: white;
  font-size: 32px;
  gap: 16px;
}

.history-card:hover .history-overlay {
  opacity: 1;
}

.overlay-icon {
  pointer-events: none;
}

.history-info {
  padding: 12px;
}

.history-prompt {
  font-size: 12px;
  color: var(--el-text-color-primary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  line-height: 1.4;
  height: 34px;
  margin-bottom: 8px;
}

.history-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
  gap: 8px;
  flex-wrap: wrap;
}

/* Pending Task Card Styles */
.pending-task-card {
  border: 2px solid var(--el-color-primary);
  animation: pulse-border 2s infinite;
}

.pending-task-card.interrupted {
  border-color: var(--el-color-warning);
  animation: none;
}

@keyframes pulse-border {
  0%, 100% { border-color: var(--el-color-primary); }
  50% { border-color: var(--el-color-primary-light-3); }
}

.pending-thumb {
  background: linear-gradient(135deg, var(--el-bg-color-page), var(--el-bg-color));
}

.pending-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(0, 0, 0, 0.3);
}

.pending-icon {
  font-size: 48px;
  color: var(--el-color-primary);
}

.interrupted-icon {
  color: var(--el-color-warning);
}

.pending-text {
  font-size: 14px;
  color: var(--el-text-color-primary);
  font-weight: 500;
}

.empty-history {
  grid-column: 1 / -1;
  text-align: center;
  padding: 40px;
  color: var(--el-text-color-secondary);
  background: var(--el-fill-color-light);
  border-radius: 8px;
}

/* Lightbox Styles */
.lightbox-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.9);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.lightbox-content {
  width: 100%;
  height: 100%;
  max-width: 1400px;
  display: flex;
  flex-direction: column;
  background: #111;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 0 50px rgba(0,0,0,0.5);
}

.lightbox-header {
  padding: 16px 24px;
  background: #1a1a1a;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #333;
}

.lightbox-title {
  font-size: 18px;
  font-weight: bold;
  color: #fff;
}

.lightbox-actions {
  display: flex;
  gap: 12px;
}

.lightbox-image-container {
  flex: 1;
  position: relative;
  overflow: hidden;
  background-image: 
    linear-gradient(45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(-45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #1a1a1a 75%), 
    linear-gradient(-45deg, transparent 75%, #1a1a1a 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  background-color: #000;
  cursor: grab;
}

.lightbox-image-container:active {
  cursor: grabbing;
}

.lightbox-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  pointer-events: none;
}

.lightbox-info {
  padding: 20px 24px;
  background: #1a1a1a;
  border-top: 1px solid #333;
  color: #ccc;
}

.info-row {
  display: flex;
  gap: 40px;
  margin-top: 12px;
}

.info-item {
  display: flex;
  gap: 8px;
}

.info-item .label {
  color: #888;
  font-weight: bold;
}

.info-item .text {
  color: #eee;
  font-family: monospace;
}

@media (max-width: 1024px) {
  .generation-grid {
    grid-template-columns: 1fr;
  }
  
  .params-card {
    position: static;
  }
}
</style>
