<template>
  <div class="training-page">
    <div class="page-header">
      <h1 class="gradient-text">开始训练</h1>
      <p class="subtitle">启动 Z-Image LoRA 训练</p>
    </div>

    <!-- 训练状态 - 可点击开始/停止训练 -->
    <div 
      class="training-status glass-card" 
      :class="[statusClass, { clickable: !isRunning && !isStarting && !isLoading }]"
      @click="handleStatusClick"
    >
      <div class="status-indicator">
        <div class="pulse-ring" v-if="isRunning || isLoading"></div>
        <el-icon :size="48">
          <Loading v-if="isRunning || isStarting || isLoading" class="spin" />
          <VideoPlay v-else-if="!isRunning && !hasCompleted" />
          <SuccessFilled v-else />
        </el-icon>
      </div>
      <div class="status-info">
        <h2>{{ statusText }}</h2>
        <p v-if="isLoading">正在加载 Transformer、VAE、Text Encoder...</p>
        <p v-else-if="isRunning">
          Epoch {{ progress.currentEpoch }}/{{ progress.totalEpochs }} · 
          Step {{ progress.currentStep }}/{{ progress.totalSteps }}
        </p>
        <p v-else-if="isStarting">正在启动...</p>
        <p v-else>点击此处开始训练（请先确认配置参数）</p>
      </div>
      <div class="status-progress" v-if="isRunning && !isLoading">
        <el-progress
          :percentage="trainingStore.progressPercent"
          :stroke-width="12"
          :show-text="true"
          :format="() => `${trainingStore.progressPercent}%`"
        />
      </div>
    </div>

    <!-- 训练信息卡片 -->
    <div class="info-cards" v-if="isRunning">
      <div class="info-card glass-card">
        <div class="card-label">当前 Loss</div>
        <div class="card-value">{{ progress.loss.toFixed(4) }}</div>
      </div>
      <div class="info-card glass-card">
        <div class="card-label">学习率</div>
        <div class="card-value">{{ progress.learningRate.toExponential(2) }}</div>
      </div>
      <div class="info-card glass-card">
        <div class="card-label">已用时间</div>
        <div class="card-value">{{ formatTime(progress.elapsedTime) }}</div>
      </div>
      <div class="info-card glass-card">
        <div class="card-label">预计剩余</div>
        <div class="card-value">{{ formatTime(progress.estimatedTimeRemaining) }}</div>
      </div>
    </div>

    <!-- 配置预览 -->
    <div class="config-preview glass-card" v-if="currentConfig">
      <div class="preview-header">
        <h3>训练配置预览</h3>
        <span class="config-name-badge">{{ currentConfig.name }}</span>
        <span class="edit-link" @click="goToEditConfig">
          <el-icon><Edit /></el-icon>
          编辑
        </span>
      </div>
      
      <!-- AC-RF 参数 -->
      <div class="preview-section">
        <h4>AC-RF 参数</h4>
        <div class="preview-grid-3">
          <div class="preview-item">
            <span class="label">Turbo Steps</span>
            <span class="value">{{ currentConfig.acrf?.turbo_steps ?? 10 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Shift</span>
            <span class="value">{{ currentConfig.acrf?.shift ?? 3.0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Jitter Scale</span>
            <span class="value">{{ currentConfig.acrf?.jitter_scale ?? 0.02 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">SNR Gamma</span>
            <span class="value">{{ currentConfig.acrf?.snr_gamma ?? 5.0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">SNR Floor</span>
            <span class="value">{{ currentConfig.acrf?.snr_floor ?? 0.1 }}</span>
          </div>
        </div>
      </div>
      
      <!-- LoRA 配置 -->
      <div class="preview-section">
        <h4>LoRA 配置</h4>
        <div class="preview-grid-3">
          <div class="preview-item">
            <span class="label">Network Dim</span>
            <span class="value">{{ currentConfig.network?.dim ?? 8 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Network Alpha</span>
            <span class="value">{{ currentConfig.network?.alpha ?? 4 }}</span>
          </div>
        </div>
      </div>
      
      <!-- 训练设置 -->
      <div class="preview-section">
        <h4>训练设置</h4>
        <div class="preview-grid-3">
          <div class="preview-item">
            <span class="label">输出名称</span>
            <span class="value highlight">{{ currentConfig.training?.output_name || 'zimage-lora' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">训练轮数</span>
            <span class="value">{{ currentConfig.advanced?.num_train_epochs ?? 10 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">保存间隔</span>
            <span class="value">每 {{ currentConfig.advanced?.save_every_n_epochs ?? 1 }} 轮</span>
          </div>
          <div class="preview-item">
            <span class="label">优化器</span>
            <span class="value">{{ currentConfig.optimizer?.type || 'AdamW8bit' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">学习率</span>
            <span class="value">{{ currentConfig.training?.learning_rate ?? 0.0001 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Weight Decay</span>
            <span class="value">{{ currentConfig.training?.weight_decay ?? 0.01 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">调度器</span>
            <span class="value">{{ currentConfig.training?.lr_scheduler || 'constant' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Warmup Steps</span>
            <span class="value">{{ currentConfig.training?.lr_warmup_steps ?? 0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Lambda FFT</span>
            <span class="value">{{ currentConfig.training?.lambda_fft ?? 0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Lambda Cosine</span>
            <span class="value">{{ currentConfig.training?.lambda_cosine ?? 0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">损失模式</span>
            <span class="value highlight">{{ getLossModeLabel(currentConfig.training?.loss_mode) }}</span>
          </div>
        </div>
        <!-- 频域感知参数 -->
        <div class="preview-grid-3" v-if="['frequency', 'unified'].includes(currentConfig.training?.loss_mode)">
          <div class="preview-item">
            <span class="label">Alpha HF (高频)</span>
            <span class="value">{{ currentConfig.training?.alpha_hf ?? 1.0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Beta LF (低频)</span>
            <span class="value">{{ currentConfig.training?.beta_lf ?? 0.2 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Downsample Factor</span>
            <span class="value">{{ currentConfig.training?.downsample_factor ?? 4 }}</span>
          </div>
        </div>
        <!-- 风格结构参数 -->
        <div class="preview-grid-3" v-if="['style', 'unified'].includes(currentConfig.training?.loss_mode)">
          <div class="preview-item">
            <span class="label">λ Struct (结构)</span>
            <span class="value">{{ currentConfig.training?.lambda_struct ?? 1.0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">λ Light (光影)</span>
            <span class="value">{{ currentConfig.training?.lambda_light ?? 0.5 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">λ Color (色调)</span>
            <span class="value">{{ currentConfig.training?.lambda_color ?? 0.3 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">λ Tex (质感)</span>
            <span class="value">{{ currentConfig.training?.lambda_tex ?? 0.5 }}</span>
          </div>
        </div>
      </div>
      
      <!-- 数据集配置 -->
      <div class="preview-section">
        <h4>数据集配置</h4>
        <div class="preview-grid-3">
          <div class="preview-item">
            <span class="label">批大小</span>
            <span class="value">{{ currentConfig.dataset?.batch_size ?? 1 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">打乱数据</span>
            <span class="value">{{ currentConfig.dataset?.shuffle ? '是' : '否' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">启用分桶</span>
            <span class="value">{{ currentConfig.dataset?.enable_bucket ? '是' : '否' }}</span>
          </div>
        </div>
        <div class="datasets-list" v-if="currentConfig.dataset?.datasets?.length > 0">
          <div v-for="(ds, idx) in currentConfig.dataset.datasets" :key="idx" class="dataset-tag">
            <span class="ds-path">{{ getDatasetName(ds.cache_directory) }}</span>
            <span class="ds-repeat">×{{ ds.num_repeats || 1 }}</span>
          </div>
        </div>
        <div class="no-datasets" v-else>
          <span>⚠️ 未配置数据集</span>
        </div>
      </div>
      
      <!-- 高级选项 -->
      <div class="preview-section">
        <h4>高级选项</h4>
        <div class="preview-grid-3">
          <div class="preview-item">
            <span class="label">混合精度</span>
            <span class="value">{{ currentConfig.advanced?.mixed_precision || 'bf16' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">梯度累积</span>
            <span class="value">{{ currentConfig.advanced?.gradient_accumulation_steps ?? 4 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">Max Grad Norm</span>
            <span class="value">{{ currentConfig.advanced?.max_grad_norm ?? 1.0 }}</span>
          </div>
          <div class="preview-item">
            <span class="label">梯度检查点</span>
            <span class="value">{{ currentConfig.advanced?.gradient_checkpointing ? '是' : '否' }}</span>
          </div>
          <div class="preview-item">
            <span class="label">随机种子</span>
            <span class="value">{{ currentConfig.advanced?.seed ?? 42 }}</span>
          </div>
        </div>
      </div>
    </div>
    <div class="config-preview glass-card" v-else>
      <h3>训练配置预览</h3>
      <p class="no-config">暂无配置，请先在配置页面设置</p>
      <router-link to="/config" class="edit-link">
        <el-icon><Edit /></el-icon>
        去配置
      </router-link>
    </div>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <el-button
        v-if="isRunning"
        type="danger"
        size="large"
        @click="stopTraining"
        class="stop-button"
      >
        <el-icon><VideoPause /></el-icon>
        停止训练
      </el-button>
      
      <el-button size="large" @click="goToMonitor">
        <el-icon><DataLine /></el-icon>
        查看监控
      </el-button>
    </div>

    <!-- 日志输出 -->
    <div class="log-output glass-card">
      <div class="log-header">
        <h3>训练日志</h3>
        <el-button size="small" text @click="clearLogs">清空</el-button>
      </div>
      <div class="log-content" ref="logContainer">
        <div 
          v-for="(log, index) in logs" 
          :key="index"
          class="log-line"
          :class="log.level"
        >
          <span class="log-time">{{ log.time }}</span>
          <span class="log-message">{{ log.message }}</span>
        </div>
        <div v-if="logs.length === 0" class="log-empty">
          等待训练开始...
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import { useWebSocketStore } from '@/stores/websocket'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Loading, VideoPlay, VideoPause, SuccessFilled, Edit, DataLine } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const trainingStore = useTrainingStore()
const wsStore = useWebSocketStore()

const isStarting = ref(false)
const hasCompleted = ref(false)
const logContainer = ref<HTMLElement>()

// 使用 wsStore 的日志
const logs = computed(() => wsStore.logs)

// 从后端加载的当前配置（原始结构，不转换）
const currentConfig = ref<any>(null)

const progress = computed(() => trainingStore.progress)
const isRunning = computed(() => trainingStore.progress.isRunning)
const isLoading = computed(() => trainingStore.progress.isLoading)

// 加载当前配置
async function loadCurrentConfig() {
  try {
    const res = await axios.get('/api/training/config/current')
    currentConfig.value = res.data
    console.log('Loaded config:', res.data)
  } catch (e) {
    console.error('Failed to load current config:', e)
  }
}

// 跳转到配置页面编辑当前配置
function goToEditConfig() {
  const configName = currentConfig.value?.name || 'default'
  router.push({ path: '/config', query: { edit: configName } })
}

const statusClass = computed(() => ({
  running: isRunning.value && !isLoading.value,
  loading: isLoading.value,
  completed: hasCompleted.value && !isRunning.value
}))

const statusText = computed(() => {
  if (isStarting.value) return '正在启动...'
  if (isLoading.value) return '模型加载中...'
  if (isRunning.value) return '训练进行中'
  if (hasCompleted.value) return '训练已完成'
  return '准备就绪'
})

function handleStatusClick() {
  if (isRunning.value || isStarting.value) return
  startTraining()
}

function goToMonitor() {
  router.push('/monitor')
}

function getDatasetName(path: string): string {
  if (!path) return '未知'
  // 提取路径最后一部分作为名称
  const parts = path.replace(/\\/g, '/').split('/')
  return parts[parts.length - 1] || parts[parts.length - 2] || path
}

function getLossModeLabel(mode: string | undefined): string {
  const labels: Record<string, string> = {
    'standard': 'Standard (基础)',
    'frequency': 'Frequency (频域感知)',
    'style': 'Style (风格结构)',
    'unified': 'Unified (统一模式)'
  }
  return labels[mode || 'standard'] || 'Standard (基础)'
}

function formatTime(seconds: number): string {
  if (seconds <= 0) return '--:--:--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

function addLog(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  // 使用 wsStore 添加日志
  wsStore.addLog(message, type)
  
  // 滚动到底部
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}

function clearLogs() {
  wsStore.clearLogs()
}

async function startTraining() {
  try {
    await ElMessageBox.confirm(
      '⚠️ 请确认以下参数后再开始训练：\n\n' +
      '• 数据集路径是否正确\n' +
      '• 训练轮数 (Epochs) 是否合适\n' +
      '• 学习率和调度器设置\n' +
      '• LoRA 参数 (Rank/Alpha)\n\n' +
      '配置可在「训练配置」页面修改。',
      '确认开始训练',
      { confirmButtonText: '确认开始', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return
  }
  
  // 确保配置已加载
  if (!currentConfig.value) {
    await loadCurrentConfig()
  }
  
  if (!currentConfig.value) {
    ElMessage.error('配置加载失败')
    return
  }
  
  isStarting.value = true
  hasCompleted.value = false
  
  try {
    addLog('正在启动 AC-RF 训练...', 'info')
    const response = await axios.post('/api/training/start', currentConfig.value)
    
    // 检查是否需要先生成缓存
    if (response.data.needs_cache) {
      addLog(`缓存不完整: Latent ${response.data.latent_cached}/${response.data.total_images}, Text ${response.data.text_cached}/${response.data.total_images}`, 'warning')
      
      try {
        await ElMessageBox.confirm(
          `数据集缓存不完整：\n` +
          `- Latent: ${response.data.latent_cached}/${response.data.total_images}\n` +
          `- Text: ${response.data.text_cached}/${response.data.total_images}\n\n` +
          `是否自动生成缓存？完成后将自动开始训练。`,
          '缓存不完整',
          { confirmButtonText: '自动生成', cancelButtonText: '取消', type: 'warning' }
        )
        
        // 自动生成缓存并重试
        await generateCacheAndStartTraining()
        
      } catch {
        addLog('用户取消了缓存生成', 'info')
        isStarting.value = false
      }
      return
    }
    
    trainingStore.progress.isRunning = true
    addLog('AC-RF 训练已启动', 'success')
  } catch (error: any) {
    addLog(`启动失败: ${error.response?.data?.detail || error.message}`, 'error')
    ElMessage.error('启动训练失败')
    isStarting.value = false
  }
}

async function generateCacheAndStartTraining() {
  if (!currentConfig.value) return
  
  const datasets = currentConfig.value.dataset?.datasets || []
  if (datasets.length === 0) {
    addLog('没有配置数据集', 'error')
    isStarting.value = false
    return
  }
  
  addLog('开始自动生成缓存...', 'info')
  
  try {
    // 对每个数据集生成缓存
    for (const ds of datasets) {
      const datasetPath = ds.cache_directory
      if (!datasetPath) continue
      
      addLog(`正在为 ${datasetPath} 生成缓存...`, 'info')
      
      // 获取绝对路径（从后端默认配置）
      const defaultsRes = await axios.get('/api/training/defaults')
      const vaePath = defaultsRes.data.vaePath
      const textEncoderPath = defaultsRes.data.textEncoderPath
      
      await axios.post('/api/cache/generate', {
        datasetPath: datasetPath,
        generateLatent: true,
        generateText: true,
        vaePath: vaePath,
        textEncoderPath: textEncoderPath,
        resolution: ds.resolution_limit || 1024,
        batchSize: 1
      })
    }
    
    addLog('缓存任务已启动，等待完成...', 'info')
    
    // 等待缓存完成
    await waitForCacheAndRetry()
    
  } catch (error: any) {
    addLog(`缓存生成失败: ${error.response?.data?.detail || error.message}`, 'error')
    isStarting.value = false
  }
}

async function waitForCacheAndRetry() {
  // 轮询等待缓存完成
  const maxWait = 60 * 60 * 1000 // 最长 60 分钟
  const startTime = Date.now()
  
  // 记录上次状态，用于检测变化
  let lastLatentStatus = ''
  let lastTextStatus = ''
  let lastLatentProgress = -1
  let lastTextProgress = -1
  let latentCompleted = false
  let textCompleted = false
  
  const poll = async () => {
    if (Date.now() - startTime > maxWait) {
      addLog('缓存生成超时', 'error')
      isStarting.value = false
      return
    }
    
    const status = wsStore.cacheStatus
    
    // Latent 状态变化检测
    if (status.latent.status !== lastLatentStatus) {
      lastLatentStatus = status.latent.status
      if (status.latent.status === 'running') {
        addLog('Latent 缓存生成中...', 'info')
      } else if (status.latent.status === 'completed' && !latentCompleted) {
        latentCompleted = true
        addLog('✓ Latent 缓存完成', 'success')
      } else if (status.latent.status === 'failed') {
        addLog('✗ Latent 缓存失败', 'error')
      }
    }
    
    // Latent 进度日志
    if (status.latent.status === 'running' && status.latent.current !== lastLatentProgress) {
      lastLatentProgress = status.latent.current || 0
      const total = status.latent.total || 0
      if (total > 0) {
        addLog(`[Latent] ${lastLatentProgress}/${total} (${Math.round(lastLatentProgress/total*100)}%)`, 'info')
      }
    }
    
    // Text 状态变化检测
    if (status.text.status !== lastTextStatus) {
      lastTextStatus = status.text.status
      if (status.text.status === 'running') {
        addLog('Text 缓存生成中...', 'info')
      } else if (status.text.status === 'completed' && !textCompleted) {
        textCompleted = true
        addLog('✓ Text 缓存完成', 'success')
      } else if (status.text.status === 'failed') {
        addLog('✗ Text 缓存失败', 'error')
      }
    }
    
    // Text 进度日志
    if (status.text.status === 'running' && status.text.current !== lastTextProgress) {
      lastTextProgress = status.text.current || 0
      const total = status.text.total || 0
      if (total > 0) {
        addLog(`[Text] ${lastTextProgress}/${total} (${Math.round(lastTextProgress/total*100)}%)`, 'info')
      }
    }
    
    const latentDone = status.latent.status !== 'running'
    const textDone = status.text.status !== 'running'
    
    if (latentDone && textDone) {
      // 两个都完成了
      if (status.latent.status === 'failed' || status.text.status === 'failed') {
        addLog('缓存生成有失败项', 'error')
        isStarting.value = false
        return
      }
      
      addLog('全部缓存生成完成', 'success')
      addLog('等待显存释放...', 'info')
      
      // 等待显存释放（缓存脚本会在完成后清理GPU）
      await new Promise(r => setTimeout(r, 3000))
      
      addLog('正在启动训练...', 'info')
      
      // 重新启动训练
      try {
        const response = await axios.post('/api/training/start', currentConfig.value)
        if (response.data.needs_cache) {
          addLog(`缓存仍不完整: Latent ${response.data.latent_cached}/${response.data.total_images}, Text ${response.data.text_cached}/${response.data.total_images}`, 'error')
          isStarting.value = false
          return
        }
        trainingStore.progress.isRunning = true
        addLog('AC-RF 训练已启动', 'success')
      } catch (error: any) {
        addLog(`启动失败: ${error.response?.data?.detail || error.message}`, 'error')
        isStarting.value = false
      }
      return
    }
    
    // 继续轮询
    setTimeout(poll, 2000)
  }
  
  // 延迟开始轮询
  setTimeout(poll, 1500)
}

async function stopTraining() {
  try {
    await ElMessageBox.confirm(
      '确定要停止训练吗？当前进度将保存。',
      '停止训练',
      { confirmButtonText: '停止', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return
  }
  
  try {
    addLog('正在停止训练...', 'warning')
    await trainingStore.stopTraining()
    addLog('训练已停止', 'warning')
  } catch (error: any) {
    addLog(`停止失败: ${error.message}`, 'error')
    ElMessage.error('停止训练失败')
  }
}

// WebSocket 由 App.vue 中的 wsStore 统一管理
// 训练日志通过 wsStore.logs 自动更新

// 监听日志变化，自动滚动到底部
watch(() => wsStore.logs.length, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
})

// 监听训练完成
watch(() => trainingStore.progress.isRunning, (running, wasRunning) => {
  if (wasRunning && !running) {
    hasCompleted.value = true
  }
})

onMounted(async () => {
  // 加载当前配置预览
  await loadCurrentConfig()
})

onUnmounted(() => {
  // WebSocket 由 wsStore 统一管理，无需单独处理
})
</script>

<style lang="scss" scoped>
.training-page {
  max-width: 1000px;
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
  }
}

.training-status {
  display: flex;
  align-items: center;
  gap: var(--space-xl);
  padding: var(--space-xl);
  margin-bottom: var(--space-lg);
  transition: all 0.3s ease;
  
  &.clickable {
    cursor: pointer;
    
    &:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(var(--primary-rgb), 0.2);
      
      .status-indicator {
        color: var(--primary);
      }
    }
  }
  
  &.running {
    border-color: var(--success);
    
    .status-indicator {
      color: var(--success);
    }
  }
  
  &.completed {
    border-color: var(--primary);
    
    .status-indicator {
      color: var(--primary);
    }
  }
  
  .status-indicator {
    position: relative;
    color: var(--text-muted);
    
    .pulse-ring {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 80px;
      height: 80px;
      border: 2px solid var(--success);
      border-radius: 50%;
      animation: pulse-ring 2s infinite;
    }
    
    .spin {
      animation: spin 1s linear infinite;
    }
  }
  
  .status-info {
    flex: 1;
    
    h2 {
      font-size: 1.5rem;
      margin-bottom: var(--space-xs);
    }
    
    p {
      color: var(--text-muted);
    }
  }
  
  .status-progress {
    width: 200px;
    
    :deep(.el-progress-bar__outer) {
      background: rgba(255, 255, 255, 0.1);
    }
    
    :deep(.el-progress-bar__inner) {
      background: linear-gradient(90deg, var(--primary), var(--success));
    }
  }
}

@keyframes pulse-ring {
  0% { transform: translate(-50%, -50%) scale(0.8); opacity: 1; }
  100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.info-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
  
  @media (max-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
}

.info-card {
  padding: var(--space-lg);
  text-align: center;
  
  .card-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: var(--space-sm);
  }
  
  .card-value {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary);
  }
}

.config-preview {
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
  position: relative;
  
  .preview-header {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    margin-bottom: var(--space-lg);
    
    h3 {
      margin: 0;
      color: var(--text-secondary);
    }
    
    .config-name-badge {
      background: var(--primary);
      color: white;
      padding: 2px 10px;
      border-radius: var(--radius-sm);
      font-size: 0.8rem;
      font-weight: 600;
    }
    
    .edit-link {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 4px;
      color: var(--primary);
      text-decoration: none;
      font-size: 0.85rem;
      
      &:hover {
        text-decoration: underline;
      }
    }
  }
  
  .no-config {
    color: var(--text-muted);
    text-align: center;
    padding: var(--space-lg);
  }
  
  .preview-section {
    margin-bottom: var(--space-md);
    padding-bottom: var(--space-md);
    border-bottom: 1px solid var(--border);
    
    &:last-child {
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }
    
    h4 {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: var(--space-sm);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }
  
  .preview-grid-3 {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: var(--space-sm) var(--space-md);
    
    @media (max-width: 1200px) {
      grid-template-columns: repeat(4, 1fr);
    }
    
    @media (max-width: 900px) {
      grid-template-columns: repeat(3, 1fr);
    }
    
    @media (max-width: 600px) {
      grid-template-columns: repeat(2, 1fr);
    }
  }
  
  .preview-item {
    .label {
      display: block;
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-bottom: 2px;
    }
    
    .value {
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: var(--text-secondary);
      
      &.highlight {
        color: var(--primary);
        font-weight: 600;
      }
    }
  }
  
  .datasets-list {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
    margin-top: var(--space-sm);
  }
  
  .dataset-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(var(--primary-rgb), 0.1);
    border: 1px solid rgba(var(--primary-rgb), 0.3);
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    
    .ds-path {
      color: var(--text-secondary);
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    .ds-repeat {
      color: var(--primary);
      font-weight: 600;
    }
  }
  
  .no-datasets {
    color: var(--warning);
    font-size: 0.85rem;
    padding: var(--space-sm) 0;
  }
  
  .preview-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    
    @media (max-width: 768px) {
      grid-template-columns: repeat(2, 1fr);
    }
  }
  
  .old-preview-item {
    .label {
      display: block;
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: var(--space-xs);
    }
    
    .value {
      font-family: var(--font-mono);
      font-size: 0.9rem;
    }
  }
  
  .edit-link {
    position: absolute;
    top: var(--space-lg);
    right: var(--space-lg);
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    color: var(--primary);
    text-decoration: none;
    font-size: 0.85rem;
    
    &:hover {
      text-decoration: underline;
    }
  }
}

.action-buttons {
  display: flex;
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
  
  .start-button {
    min-width: 160px;
    box-shadow: 0 0 30px var(--primary-glow);
  }
  
  .stop-button {
    min-width: 160px;
  }
}

.log-output {
  padding: var(--space-lg);
  
  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-md);
    
    h3 {
      color: var(--text-secondary);
    }
  }
  
  .log-content {
    height: 300px;
    overflow-y: auto;
    background: var(--bg-darker);
    border-radius: var(--radius-md);
    padding: var(--space-md);
    font-family: var(--font-mono);
    font-size: 0.85rem;
  }
  
  .log-line {
    display: flex;
    gap: var(--space-md);
    padding: var(--space-xs) 0;
    border-bottom: 1px solid var(--border);
    
    &:last-child {
      border-bottom: none;
    }
    
    .log-time {
      color: var(--text-muted);
      flex-shrink: 0;
    }
    
    .log-message {
      color: var(--text-secondary);
    }
    
    &.success .log-message { color: var(--success); }
    &.warning .log-message { color: var(--warning); }
    &.error .log-message { color: var(--error); }
  }
  
  .log-empty {
    color: var(--text-muted);
    text-align: center;
    padding: var(--space-xl);
  }
}
</style>

