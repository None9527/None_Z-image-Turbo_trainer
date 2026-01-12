<template>
  <div class="monitor-page">
    <div class="page-header">
      <h1 class="gradient-text">训练监控</h1>
      <p class="subtitle">实时查看训练状态和曲线</p>
      
      <!-- 训练记录选择器 - 移到标题下方 -->
      <div class="run-selector-row">
        <el-select 
          v-model="selectedRun" 
          placeholder="选择训练记录" 
          @change="loadRunData"
          :loading="loadingRuns"
          style="width: 380px"
          popper-class="run-selector-dropdown"
        >
          <el-option 
            v-for="run in availableRuns" 
            :key="run.name" 
            :label="run.name" 
            :value="run.name"
          >
            <div class="run-option">
              <span class="run-name">{{ run.name }}</span>
              <span class="run-time">{{ formatRunTime(run.start_time) }}</span>
            </div>
          </el-option>
        </el-select>
        
        <el-button-group>
          <el-button 
            type="primary"
            :icon="Aim" 
            @click="jumpToCurrentRun"
            :disabled="!currentRunName || selectedRun === currentRunName"
          >当前训练</el-button>
          <el-button 
            :icon="Refresh" 
            @click="refreshRuns"
            :loading="loadingRuns"
          />
        </el-button-group>
      </div>
    </div>

    <!-- 实时状态 -->
    <div class="realtime-stats">
      <div class="stat-card glass-card">
        <div class="stat-icon">
          <el-icon :size="28"><DataLine /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">当前 Loss</div>
          <div class="stat-value loss">{{ progress.loss.toFixed(6) }}</div>
        </div>
        <div class="stat-trend" :class="lossTrend">
          <el-icon><ArrowDown v-if="lossTrend === 'down'" /><ArrowUp v-else /></el-icon>
        </div>
      </div>

      <div class="stat-card glass-card">
        <div class="stat-icon lr">
          <el-icon :size="28"><Setting /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">学习率</div>
          <div class="stat-value lr">{{ progress.learningRate.toExponential(2) }}</div>
        </div>
      </div>

      <div class="stat-card glass-card">
        <div class="stat-icon step">
          <el-icon :size="28"><Timer /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">当前步数</div>
          <div class="stat-value">{{ progress.currentStep }} / {{ progress.totalSteps }}</div>
        </div>
      </div>

      <div class="stat-card glass-card">
        <div class="stat-icon epoch">
          <el-icon :size="28"><Refresh /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">当前 Epoch</div>
          <div class="stat-value">{{ progress.currentEpoch }} / {{ progress.totalEpochs }}</div>
        </div>
      </div>
    </div>

    <!-- Loss 曲线 -->
    <div class="chart-container glass-card">
      <div class="chart-header">
        <h3>Loss 曲线</h3>
        <div class="chart-controls">
          <div class="smoothing-control">
            <span class="label">平滑: {{ smoothing }}</span>
            <el-slider v-model="smoothing" :min="0" :max="0.99" :step="0.001" size="small" style="width: 100px; margin-right: 12px" />
          </div>
          <el-radio-group v-model="lossChartScale" size="small">
            <el-radio-button label="linear">线性</el-radio-button>
            <el-radio-button label="log">对数</el-radio-button>
          </el-radio-group>
        </div>
      </div>
      <div class="chart">
        <v-chart :option="lossChartOption" autoresize />
      </div>
    </div>

    <!-- 学习率曲线 -->
    <div class="chart-container glass-card">
      <div class="chart-header">
        <h3>学习率曲线</h3>
      </div>
      <div class="chart">
        <v-chart :option="lrChartOption" autoresize />
      </div>
    </div>

    <!-- GPU 监控 -->
    <div class="gpu-monitor glass-card">
      <div class="chart-header">
        <h3>GPU 监控</h3>
      </div>
      <div class="gpu-stats">
        <div class="gpu-stat">
          <div class="stat-label">显存使用</div>
          <el-progress 
            :percentage="systemStore.gpuInfo.memoryPercent" 
            :stroke-width="20"
            :format="(p: number) => `${p}%`"
          />
          <div class="stat-detail">
            {{ systemStore.gpuInfo.memoryUsed }} / {{ systemStore.gpuInfo.memoryTotal }} GB
          </div>
        </div>
        <div class="gpu-stat">
          <div class="stat-label">GPU 利用率</div>
          <el-progress 
            :percentage="systemStore.gpuInfo.utilization" 
            :stroke-width="20"
            :format="(p: number) => `${p}%`"
          />
        </div>
        <div class="gpu-stat">
          <div class="stat-label">温度</div>
          <div class="temperature">
            <span class="temp-value">{{ systemStore.gpuInfo.temperature }}°C</span>
            <el-icon :class="tempClass"><Sunny v-if="tempClass === 'cool'" /><Sunrise v-else /></el-icon>
          </div>
        </div>
      </div>
    </div>

    <!-- 训练时间统计 -->
    <div class="time-stats glass-card">
      <div class="chart-header">
        <h3>时间统计</h3>
      </div>
      <div class="time-grid">
        <div class="time-item">
          <div class="time-label">已运行时间</div>
          <div class="time-value">{{ formatTime(progress.elapsedTime) }}</div>
        </div>
        <div class="time-item">
          <div class="time-label">预计剩余时间</div>
          <div class="time-value">{{ formatTime(progress.estimatedTimeRemaining) }}</div>
        </div>
        <div class="time-item">
          <div class="time-label">预计完成时间</div>
          <div class="time-value">{{ estimatedEndTime }}</div>
        </div>
        <div class="time-item">
          <div class="time-label">平均步数速度</div>
          <div class="time-value">{{ avgStepTime }} s/step</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useTrainingStore } from '@/stores/training'
import { useSystemStore } from '@/stores/system'
import VChart from 'vue-echarts'
import axios from 'axios'
import { Refresh, Aim } from '@element-plus/icons-vue'

const trainingStore = useTrainingStore()
const systemStore = useSystemStore()

// 图表控制
const lossChartScale = ref<'linear' | 'log'>('linear')
const smoothing = ref(0.99)

// 训练记录选择
interface TrainingRun {
  name: string
  start_time: string
  logdir: string
  event_files: number
}

const availableRuns = ref<TrainingRun[]>([])
const selectedRun = ref('')
const loadingRuns = ref(false)
const historyLoss = ref<number[]>([])  // 从API加载的历史Loss
const historyLr = ref<number[]>([])    // 从API加载的历史LR

// 当前正在训练的记录名称（从配置获取）
const currentRunName = computed(() => {
  // 优先使用正在训练的配置名称，否则使用列表第一个
  if (trainingStore.progress.isRunning && trainingStore.config.outputName) {
    return trainingStore.config.outputName
  }
  return availableRuns.value.length > 0 ? availableRuns.value[0].name : ''
})

// 跳转到当前训练
async function jumpToCurrentRun() {
  const targetName = currentRunName.value
  if (!targetName) return
  
  // 先刷新列表确保包含最新记录
  await refreshRuns()
  
  // 在列表中查找匹配的记录
  const found = availableRuns.value.find(r => r.name === targetName)
  if (found) {
    selectedRun.value = targetName
    await loadRunData()
  }
}

// 获取训练记录列表
async function refreshRuns() {
  loadingRuns.value = true
  try {
    const res = await axios.get('/api/training/runs')
    availableRuns.value = res.data.runs || []
    
    // 如果有记录且未选择，自动选择最新的
    if (availableRuns.value.length > 0 && !selectedRun.value) {
      selectedRun.value = availableRuns.value[0].name
      await loadRunData()
    }
  } catch (e) {
    console.error('获取训练记录失败:', e)
  } finally {
    loadingRuns.value = false
  }
}

// 加载指定训练记录的数据
async function loadRunData() {
  if (!selectedRun.value) return
  
  try {
    const res = await axios.get('/api/training/all-scalars', {
      params: { run: selectedRun.value }
    })
    
    const scalars = res.data.scalars || {}
    
    // 提取 loss 数据
    const lossData = scalars['train/loss'] || scalars['train/ema_loss'] || []
    historyLoss.value = lossData.map((d: any) => d.value)
    
    // 提取 lr 数据
    const lrData = scalars['train/learning_rate'] || []
    historyLr.value = lrData.map((d: any) => d.value)
  } catch (e) {
    console.error('加载训练数据失败:', e)
  }
}

// 格式化运行时间
function formatRunTime(isoTime: string): string {
  try {
    const date = new Date(isoTime)
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

// 轮询定时器（训练中时定时刷新TensorBoard数据）
let pollTimer: ReturnType<typeof setInterval> | null = null
const POLL_INTERVAL = 5000  // 5秒轮询一次

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    // 从当前训练对应的日志加载最新数据
    await loadRunData()
  }, POLL_INTERVAL)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// 监听训练状态，控制轮询
watch(() => trainingStore.progress.isRunning, (running) => {
  if (running) {
    // 训练开始时，刷新训练记录列表并开始轮询
    refreshRuns()
    startPolling()
  } else {
    // 训练结束，停止轮询并刷新最终数据
    stopPolling()
    refreshRuns()
  }
}, { immediate: true })

// 页面加载时获取训练记录
onMounted(() => {
  refreshRuns()
})

// 页面卸载时清理定时器
onUnmounted(() => {
  stopPolling()
})

// GPU 数据通过 WebSocket 实时更新到 systemStore，无需轮询

const progress = computed(() => trainingStore.progress)

const lossTrend = computed(() => {
  const history = progress.value.lossHistory
  if (history.length < 2) return 'neutral'
  return history[history.length - 1] < history[history.length - 2] ? 'down' : 'up'
})

const tempClass = computed(() => {
  const temp = systemStore.gpuInfo.temperature
  if (temp < 60) return 'cool'
  if (temp < 80) return 'warm'
  return 'hot'
})

const estimatedEndTime = computed(() => {
  if (progress.value.estimatedTimeRemaining <= 0) return '--:--'
  const endDate = new Date(Date.now() + progress.value.estimatedTimeRemaining * 1000)
  return endDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
})

const avgStepTime = computed(() => {
  if (progress.value.currentStep === 0) return '--'
  return (progress.value.elapsedTime / progress.value.currentStep).toFixed(2)
})

const baseChartConfig = {
  backgroundColor: 'transparent',
  grid: {
    top: 20,
    right: 40,
    bottom: 40,
    left: 60
  },
  xAxis: {
    type: 'category' as const,
    axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
    axisLabel: { color: 'rgba(255,255,255,0.5)' },
    splitLine: { show: false }
  },
  yAxis: {
    type: 'value' as const,
    axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
    axisLabel: { color: 'rgba(255,255,255,0.5)' },
    splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
  },
  tooltip: {
    trigger: 'axis' as const,
    backgroundColor: 'rgba(10,10,15,0.9)',
    borderColor: 'rgba(0,245,255,0.3)',
    textStyle: { color: '#fff' }
  }
}

const itemStyle = {
  normal: {
    color: '#00f5ff',
    lineStyle: {
      color: '#00f5ff',
      width: 1
    }
  }
}

// 图表数据直接使用TensorBoard API数据
const currentLossData = computed(() => historyLoss.value)

const currentLrData = computed(() => historyLr.value)

const smoothedLoss = computed(() => {
  const data = currentLossData.value
  const smoothWeight = smoothing.value
  if (data.length === 0) return []
  
  // TensorBoard-compatible EMA with Bias Correction
  // Formula: smoothed[i] = (data[i]*(1-w) + last*w) / (1 - w^(i+1))
  let smoothedLast = 0
  const result = []
  
  for (let i = 0; i < data.length; i++) {
    smoothedLast = smoothedLast * smoothWeight + data[i] * (1 - smoothWeight)
    const debiasWeight = 1 - Math.pow(smoothWeight, i + 1)
    result.push(smoothedLast / debiasWeight)
  }
  return result
})

// TensorBoard风格的y轴范围计算（基于平滑曲线）
const lossYAxisRange = computed(() => {
  const data = smoothedLoss.value  // 使用平滑数据而非原始数据
  if (data.length === 0) return { min: 0, max: 1 }
  
  // 忽略前5%的数据（训练初期仍可能有异常）
  const skipCount = Math.max(1, Math.floor(data.length * 0.05))
  const validData = data.slice(skipCount)
  
  if (validData.length === 0) return { min: 0, max: 1 }
  
  const min = Math.min(...validData)
  const max = Math.max(...validData)
  const range = max - min
  
  // 添加10%的边距，让曲线不贴边
  return {
    min: Math.max(0, min - range * 0.1),
    max: max + range * 0.1
  }
})

const lossChartOption = computed(() => ({
  ...baseChartConfig,
  xAxis: {
    ...baseChartConfig.xAxis,
    data: currentLossData.value.map((_, i) => i + 1)
  },
  yAxis: {
    ...baseChartConfig.yAxis,
    type: lossChartScale.value === 'log' ? 'log' : 'value',
    // TensorBoard风格：自动缩放到有效数据范围
    min: lossChartScale.value === 'log' ? undefined : lossYAxisRange.value.min,
    max: lossChartScale.value === 'log' ? undefined : lossYAxisRange.value.max,
  },
  series: [
    {
      name: 'Original',
      type: 'line',
      data: currentLossData.value,
      smooth: false,
      symbol: 'none',
      itemStyle: { color: 'rgba(0, 245, 255, 0.2)' },
      lineStyle: {
        color: 'rgba(0, 245, 255, 0.2)',
        width: 1
      },
      z: 1
    },
    {
      name: 'Smoothed',
      type: 'line',
      data: smoothedLoss.value,
      smooth: true,
      symbol: 'none',
      itemStyle: { color: '#00f5ff' },
      lineStyle: {
        color: '#00f5ff',
        width: 2
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(0,245,255,0.3)' },
            { offset: 1, color: 'rgba(0,245,255,0)' }
          ]
        }
      },
      z: 2
    }
  ]
}))

const lrChartOption = computed(() => ({
  ...baseChartConfig,
  xAxis: {
    ...baseChartConfig.xAxis,
    data: currentLrData.value.map((_, i) => i + 1)
  },
  series: [
    {
      name: 'Learning Rate',
      type: 'line',
      data: currentLrData.value,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        color: '#a855f7',
        width: 2
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(168,85,247,0.3)' },
            { offset: 1, color: 'rgba(168,85,247,0)' }
          ]
        }
      }
    }
  ]
}))

function formatTime(seconds: number): string {
  if (seconds <= 0) return '--:--:--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

// GPU 数据通过 WebSocket 实时更新，无需手动获取
</script>

<style lang="scss" scoped>
.monitor-page {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: var(--space-xl);
  
  h1 {
    font-family: var(--font-display);
    font-size: 2rem;
    margin-bottom: var(--space-xs);
  }
  
  .subtitle {
    color: var(--text-muted);
    margin-bottom: var(--space-md);
  }
  
  .run-selector-row {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    margin-top: var(--space-sm);
    
    .el-select {
      :deep(.el-input__wrapper) {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
    }
    
    .run-option {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
    }
    
    .el-button-group {
      .el-button--primary {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border: none;
      }
    }
  }
}

.realtime-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
  
  @media (max-width: 1024px) {
    grid-template-columns: repeat(2, 1fr);
  }
}

.stat-card {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-lg);
  
  .stat-icon {
    width: 56px;
    height: 56px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(0,245,255,0.05));
    color: var(--primary);
    
    &.lr {
      background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(168,85,247,0.05));
      color: var(--secondary);
    }
    
    &.step {
      background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05));
      color: var(--success);
    }
    
    &.epoch {
      background: linear-gradient(135deg, rgba(244,63,94,0.2), rgba(244,63,94,0.05));
      color: var(--accent);
    }
  }
  
  .stat-content {
    flex: 1;
    
    .stat-label {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: var(--space-xs);
    }
    
    .stat-value {
      font-family: var(--font-mono);
      font-size: 1.25rem;
      font-weight: 600;
      
      &.loss {
        color: var(--primary);
      }
      
      &.lr {
        color: var(--secondary);
      }
    }
  }
  
  .stat-trend {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    
    &.down {
      background: rgba(34, 197, 94, 0.2);
      color: var(--success);
    }
    
    &.up {
      background: rgba(244, 63, 94, 0.2);
      color: var(--accent);
    }
  }
}

.chart-container {
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
  
  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-md);
    
    .chart-controls {
      display: flex;
      align-items: center;
      gap: 16px;
      
      .smoothing-control {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-muted);
        font-size: 0.85rem;
        
        .label {
          white-space: nowrap;
        }
      }
    }

    h3 {
      color: var(--text-secondary);
    }
  }
  
  .chart {
    height: 300px;
  }
}

.gpu-monitor {
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
  
  .chart-header {
    margin-bottom: var(--space-lg);
    
    h3 {
      color: var(--text-secondary);
    }
  }
  
  .gpu-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-xl);
    
    @media (max-width: 768px) {
      grid-template-columns: 1fr;
    }
  }
  
  .gpu-stat {
    .stat-label {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: var(--space-sm);
    }
    
    .stat-detail {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: var(--space-sm);
      text-align: center;
    }
    
    .temperature {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-sm);
      
      .temp-value {
        font-family: var(--font-mono);
        font-size: 2rem;
        font-weight: 600;
      }
      
      .el-icon {
        font-size: 32px;
        
        &.cool { color: var(--primary); }
        &.warm { color: var(--warning); }
        &.hot { color: var(--error); }
      }
    }
    
    :deep(.el-progress-bar__outer) {
      background: rgba(255, 255, 255, 0.1);
    }
    
    :deep(.el-progress-bar__inner) {
      background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
  }
}

.time-stats {
  padding: var(--space-lg);
  
  .chart-header {
    margin-bottom: var(--space-lg);
    
    h3 {
      color: var(--text-secondary);
    }
  }
  
  .time-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-lg);
    
    @media (max-width: 768px) {
      grid-template-columns: repeat(2, 1fr);
    }
  }
  
  .time-item {
    text-align: center;
    
    .time-label {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: var(--space-sm);
    }
    
    .time-value {
      font-family: var(--font-mono);
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--primary);
    }
  }
}
</style>

<!-- 全局样式：下拉选项需要穿透scoped -->
<style lang="scss">
.run-selector-dropdown {
  .el-select-dropdown__item {
    padding: 8px 16px;
    
    .run-option {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      gap: 24px;
      
      .run-name {
        font-weight: 500;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      
      .run-time {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        white-space: nowrap;
        flex-shrink: 0;
      }
    }
  }
}
</style>
