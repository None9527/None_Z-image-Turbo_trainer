<template>
  <div class="welcome-page">
    <!-- 左侧：品牌 + 快速入口 -->
    <div class="left-panel">
      <!-- 品牌区域 -->
      <a href="https://github.com/None9527/None_Z-image-Turbo_trainer" target="_blank" class="brand-link">
        <div class="brand">
          <div class="logo">
            <span>N</span>
          </div>
          <div class="brand-text">
            <h1><span class="gradient">None</span> Trainer</h1>
            <p class="subtitle">Z-Image Turbo LoRA 训练工作室</p>
          </div>
        </div>
      </a>
      
      <p class="description">
        基于 <strong>AC-RF</strong>（锚点耦合整流流）算法的高效 LoRA 微调工具，
        支持 10 步快速推理、自动硬件优化、实时训练监控。
      </p>

      <!-- 快速入口 -->
      <div class="nav-grid">
        <div class="nav-card" @click="$router.push('/dataset')">
          <div class="nav-icon blue"><el-icon><Picture /></el-icon></div>
          <div class="nav-content">
            <h3>数据集管理</h3>
            <p>导入图片、生成缓存、AI 标注</p>
          </div>
          <el-icon class="nav-arrow"><ArrowRight /></el-icon>
        </div>
        
        <div class="nav-card" @click="$router.push('/config')">
          <div class="nav-icon green"><el-icon><Setting /></el-icon></div>
          <div class="nav-content">
            <h3>训练配置</h3>
            <p>AC-RF 参数、LoRA、优化器</p>
          </div>
          <el-icon class="nav-arrow"><ArrowRight /></el-icon>
        </div>
        
        <div class="nav-card" @click="$router.push('/training')">
          <div class="nav-icon gold"><el-icon><VideoPlay /></el-icon></div>
          <div class="nav-content">
            <h3>开始训练</h3>
            <p>实时 Loss 曲线、进度监控</p>
          </div>
          <el-icon class="nav-arrow"><ArrowRight /></el-icon>
        </div>
        
        <div class="nav-card" @click="$router.push('/generation')">
          <div class="nav-icon orange"><el-icon><MagicStick /></el-icon></div>
          <div class="nav-content">
            <h3>图像生成</h3>
            <p>测试训练好的 LoRA 模型</p>
          </div>
          <el-icon class="nav-arrow"><ArrowRight /></el-icon>
        </div>
      </div>

      <!-- 广告位 -->
      <div class="ad-banner">
        <div class="ad-content">
          <div class="ad-glow"></div>
          <div class="ad-text">
            <span class="ad-icon">📢</span>
            <span class="ad-main">广告位招租</span>
            <span class="ad-sub">由于没想好打什么广告...</span>
          </div>
          <div class="ad-particles">
            <span v-for="i in 12" :key="i" class="particle" :style="{ '--i': i }"></span>
          </div>
        </div>
      </div>

      <!-- 底部信息 -->
      <div class="footer-info">
        <div class="tech-tags">
          <span class="tech-tag">🎯 锚点耦合采样</span>
          <span class="tech-tag">📉 Min-SNR 加权</span>
          <span class="tech-tag">⚡ Flash Attention</span>
          <span class="tech-tag">🔧 硬件自适应</span>
        </div>
        <div class="author">Made with ❤️ by <strong>None</strong></div>
      </div>
    </div>

    <!-- 右侧：状态面板 -->
    <div class="right-panel">
      <!-- 系统状态 -->
      <div class="status-card">
        <div class="card-header">
          <el-icon><Monitor /></el-icon>
          <span>系统状态</span>
          <el-tag :type="wsConnected ? 'success' : 'danger'" size="small" effect="plain">
            {{ wsConnected ? '在线' : '离线' }}
          </el-tag>
        </div>
        <div class="status-list" v-if="hasSystemInfo">
          <div class="status-row">
            <span class="label">Python</span>
            <span class="value">{{ systemInfo.python }}</span>
          </div>
          <div class="status-row">
            <span class="label">PyTorch</span>
            <span class="value">{{ systemInfo.pytorch }}</span>
          </div>
          <div class="status-row">
            <span class="label">CUDA</span>
            <span class="value">{{ systemInfo.cuda }}</span>
          </div>
          <div class="status-row">
            <span class="label">Diffusers</span>
            <span class="value">{{ systemInfo.diffusers }}</span>
          </div>
        </div>
        <div v-else class="loading-state">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>连接中...</span>
        </div>
      </div>

      <!-- 模型状态（使用抽象层接口） -->
      <div class="status-card model-card">
        <div class="card-header">
          <el-icon><Box /></el-icon>
          <span>基础模型</span>
          <el-tag :type="currentModelStatus.exists ? 'success' : 'warning'" size="small" effect="dark">
            {{ currentModelStatus.exists ? '就绪' : '需下载' }}
          </el-tag>
        </div>

        <!-- 模型类型选择 -->
        <div class="model-selector">
          <div 
            v-for="model in modelTypes" 
            :key="model.value"
            :class="['model-type-btn', { active: selectedModelType === model.value }]"
            @click="selectModelType(model.value)"
          >
            <span class="model-icon">{{ model.icon }}</span>
            <span class="model-label">{{ model.label }}</span>
          </div>
        </div>
        
        <div class="model-status" v-if="currentModelStatus.summary">
          <div class="model-ring">
            <svg viewBox="0 0 100 100">
              <circle class="ring-bg" cx="50" cy="50" r="42" />
              <circle class="ring-progress" cx="50" cy="50" r="42" :style="{ strokeDashoffset: progressOffset }" />
            </svg>
            <div class="ring-text">
              <span class="ring-num">{{ validPercent }}</span>
              <span class="ring-label">%</span>
            </div>
          </div>
          
          <div class="model-details">
            <div class="detail-row">
              <span>有效组件</span>
              <strong class="success">{{ currentModelStatus.summary.valid_components }}</strong>
            </div>
            <div class="detail-row">
              <span>总组件</span>
              <strong>{{ currentModelStatus.summary.total_components }}</strong>
            </div>
          </div>
        </div>

        <div class="component-grid" v-if="currentModelStatus.details">
          <div 
            v-for="(comp, name) in currentModelStatus.details" 
            :key="name"
            class="comp-item"
            :class="{ valid: comp.valid, missing: !comp.exists }"
          >
            <el-icon>
              <CircleCheck v-if="comp.valid" />
              <Close v-else />
            </el-icon>
            <span>{{ getComponentName(name) }}</span>
          </div>
        </div>

        <!-- 模型路径显示 -->
        <div class="model-path-info" v-if="currentModelStatus.path">
          <span class="path-label">配置路径:</span>
          <code class="path-value" :title="currentModelStatus.path">{{ currentModelStatus.path }}</code>
        </div>

        <!-- 未检测状态 -->
        <div v-if="!currentModelStatus.summary && !loadingModel" class="model-unchecked">
          <el-icon><WarningFilled /></el-icon>
          <span>未检测到模型，请配置 .env 中的模型路径</span>
        </div>

        <div v-if="loadingModel" class="loading-state">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>检测中...</span>
        </div>

        <!-- 下载按钮：完全缺失或部分缺失时显示（且没有下载进行中） -->
        <el-button 
          v-if="needsDownload && !isDownloading" 
          type="primary" 
          @click="startDownload" 
          :loading="startingDownload"
          class="download-btn"
        >
          <el-icon><Download /></el-icon>
          {{ downloadButtonText }}
        </el-button>
        
        <!-- 下载进度（总进度）：始终显示，用模型名称区分 -->
        <div v-if="isDownloading || isDownloadCompleted" class="download-progress-box">
          <div class="progress-header">
            <span>{{ isDownloadCompleted ? '下载完成' : `正在下载 ${downloadingModelName || currentModelName}` }}</span>
            <span class="progress-percent">{{ isDownloadCompleted ? 100 : downloadProgress.toFixed(1) }}%</span>
          </div>
          <el-progress 
            :percentage="isDownloadCompleted ? 100 : downloadProgress" 
            :stroke-width="10"
            :show-text="false"
            status="success"
          />
          <div class="progress-info" v-if="!isDownloadCompleted">
            <span>{{ downloadedSize }} / {{ totalSize }}</span>
            <span>{{ downloadSpeed }}</span>
          </div>
          <div class="progress-info" v-else>
             <span>已完成</span>
             <el-button link type="primary" @click="systemStore.updateDownloadStatus({ status: 'idle' })">关闭</el-button>
          </div>
        </div>
      </div>

      <!-- 联系方式 -->
      <div class="contact-card">
        <div class="contact-row" @click="copyEmail('lihaonan1082@gmail.com')">
          <span class="contact-icon">📧</span>
          <span class="contact-text">lihaonan1082@gmail.com</span>
          <el-icon class="copy-icon"><CopyDocument /></el-icon>
        </div>
        <div class="contact-row" @click="copyEmail('592532681@qq.com')">
          <span class="contact-icon">📮</span>
          <span class="contact-text">592532681@qq.com</span>
          <el-icon class="copy-icon"><CopyDocument /></el-icon>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { 
  Picture, Setting, VideoPlay, Monitor,
  CircleCheck, Close, Loading, Box,
  ArrowRight, MagicStick, Download, CopyDocument, WarningFilled
} from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const router = useRouter()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

// 多模型支持 (仅 Z-Image)
const modelTypes = ref([
  { value: 'zimage', label: 'Z-Image', icon: '⚡' }
])

const selectedModelType = ref('zimage')
const loadingModel = ref(false)

// 每个模型的状态
const modelStatusMap = ref<Record<string, any>>({
  zimage: { exists: false, details: null, summary: null, path: '' }
})

const currentModelStatus = computed(() => modelStatusMap.value[selectedModelType.value] || { exists: false, details: null, summary: null })

const currentModelName = computed(() => {
  return 'Z-Image-Turbo'
})

const startingDownload = ref(false)



// 从后端状态获取正在下载的模型信息
const downloadingModelName = computed(() => downloadStatus.value.model_name || '')
const downloadingModelType = computed(() => downloadStatus.value.model_type || '')

const systemInfo = computed(() => systemStore.systemInfo)
const wsConnected = computed(() => wsStore.isConnected)
const hasSystemInfo = computed(() => systemStore.systemInfo.python !== '')

// 下载状态
const downloadStatus = computed(() => systemStore.downloadStatus)
const isDownloading = computed(() => downloadStatus.value.status === 'running')
const isDownloadCompleted = computed(() => downloadStatus.value.status === 'completed')
const downloadProgress = computed(() => downloadStatus.value.progress || 0)
const downloadedSize = computed(() => {
  const gb = downloadStatus.value.downloaded_size_gb || 0
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(gb * 1024).toFixed(0)} MB`
})
const totalSize = computed(() => {
  const gb = downloadStatus.value.total_size_gb || 32  // 默认预估 32GB
  return `${gb.toFixed(0)} GB`
})
const downloadSpeed = computed(() => {
  const speed = downloadStatus.value.speed || 0
  const unit = downloadStatus.value.speed_unit || 'MB'
  return speed > 0 ? `${speed.toFixed(1)} ${unit}/s` : '计算中...'
})

// 监听下载状态变化
watch(() => downloadStatus.value.status, (newStatus, oldStatus) => {
  if (newStatus === 'completed' && oldStatus === 'running') {
    ElMessage.success('下载完成！正在刷新模型状态...')
    refreshModelStatus()
  }
})

// 是否需要下载（完全缺失或部分缺失）
const needsDownload = computed(() => {
  const status = currentModelStatus.value
  if (!status.summary) return !status.exists
  
  // 有缺失的组件就需要下载
  const { valid_components, total_components } = status.summary
  return valid_components < total_components
})

// 下载按钮文字
const downloadButtonText = computed(() => {
  const status = currentModelStatus.value
  if (!status.summary || !status.exists) {
    return `下载 ${currentModelName.value} 模型`
  }
  
  const { missing_components } = status.summary
  if (missing_components && missing_components.length > 0) {
    return `补充下载 (${missing_components.length} 个组件缺失)`
  }
  
  return `下载 ${currentModelName.value} 模型`
})


const validPercent = computed(() => {
  if (!currentModelStatus.value.summary) return 0
  const { valid_components, total_components } = currentModelStatus.value.summary
  return Math.round((valid_components / total_components) * 100)
})

const progressOffset = computed(() => {
  const circumference = 2 * Math.PI * 42
  return circumference - (validPercent.value / 100) * circumference
})

const componentNames: Record<string, string> = {
  'transformer': 'Transformer',
  'vae': 'VAE',
  'text_encoder': 'Text Encoder',
  'tokenizer': 'Tokenizer',
  'scheduler': 'Scheduler',
  'model_index.json': 'Model Index'
}

function getComponentName(name: string | number): string {
  const key = String(name)
  return componentNames[key] || key
}

async function selectModelType(type: string) {
  selectedModelType.value = type
  await refreshModelStatus(type)
}

async function refreshModelStatus(modelType?: string) {
  const type = modelType || selectedModelType.value
  loadingModel.value = true
  try {
    // 使用抽象层接口检测模型状态
    const res = await axios.get(`/api/system/model-detector/detect/${type}`)
    if (res.data.success) {
      modelStatusMap.value[type] = {
        exists: res.data.status === 'valid' || res.data.status === 'incomplete',
        details: res.data.components,
        summary: res.data.summary,
        path: res.data.path
      }
    } else {
      modelStatusMap.value[type] = { exists: false, details: null, summary: null, path: '' }
    }
  } catch (e) {
    console.error('Failed to check model status:', e)
    modelStatusMap.value[type] = { exists: false, details: null, summary: null, path: '' }
  } finally {
    loadingModel.value = false
  }
}

async function startDownload() {
  startingDownload.value = true
  try {
    const res = await axios.post(`/api/system/download-model?model_type=${selectedModelType.value}`)
    ElMessage.success(`${currentModelName.value} 下载任务已启动`)
  } catch (e: any) {
    ElMessage.error('启动下载失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    startingDownload.value = false
  }
}

function copyEmail(email: string) {
  navigator.clipboard.writeText(email)
  ElMessage.success(`已复制: ${email}`)
}

refreshModelStatus()
</script>

<style scoped>
.welcome-page {
  height: 100%;
  display: flex;
  gap: 32px;
  padding: 32px;
  background: var(--bg-primary);
  overflow: hidden;
}

/* 左侧面板 */
.left-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.brand-link {
  text-decoration: none;
  display: block;
  margin-bottom: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.brand:hover {
  border-color: var(--primary);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(240, 180, 41, 0.15);
}

.brand:hover .logo {
  box-shadow: 0 8px 32px rgba(240, 180, 41, 0.4);
}

.logo {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #f0b429 0%, #e67e22 100%);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 24px rgba(240, 180, 41, 0.3);
  transition: box-shadow 0.3s;
  flex-shrink: 0;
}

.logo span {
  font-size: 36px;
  font-weight: 800;
  color: #1a1a1d;
}

.brand-text h1 {
  font-size: 2rem;
  font-weight: 800;
  margin: 0;
  letter-spacing: -1px;
  color: var(--text-primary);
}

.brand-text .gradient {
  background: linear-gradient(135deg, #f0b429 0%, #e67e22 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.brand-text .subtitle {
  margin: 4px 0 0 0;
  color: var(--text-muted);
  font-size: 14px;
}

.description {
  color: var(--text-secondary);
  line-height: 1.7;
  margin: 0 0 24px 0;
  padding: 0 4px;
}

.description strong {
  color: var(--primary);
}

/* 导航卡片 */
.nav-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}

.nav-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 18px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.nav-card:hover {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.03);
  transform: translateX(4px);
}

.nav-card:hover .nav-arrow {
  color: var(--primary);
  transform: translateX(4px);
}

.nav-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  flex-shrink: 0;
}

.nav-icon.blue { background: rgba(64, 158, 255, 0.12); color: #409eff; }
.nav-icon.green { background: rgba(103, 194, 58, 0.12); color: #67c23a; }
.nav-icon.gold { background: rgba(240, 180, 41, 0.12); color: #f0b429; }
.nav-icon.orange { background: rgba(230, 126, 34, 0.12); color: #e67e22; }

.nav-content {
  flex: 1;
  min-width: 0;
}

.nav-content h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.nav-content p {
  margin: 4px 0 0 0;
  font-size: 13px;
  color: var(--text-muted);
}

.nav-arrow {
  color: var(--text-muted);
  font-size: 16px;
  transition: all 0.2s;
}

/* 广告位 */
.ad-banner {
  margin-top: 24px;
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  border: 1px solid rgba(240, 180, 41, 0.2);
}

.ad-content {
  position: relative;
  padding: 24px 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
}

.ad-glow {
  position: absolute;
  top: 0;
  left: -100%;
  width: 200%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(240, 180, 41, 0.1) 25%,
    rgba(240, 180, 41, 0.3) 50%,
    rgba(240, 180, 41, 0.1) 75%,
    transparent 100%
  );
  animation: glow-sweep 3s ease-in-out infinite;
}

@keyframes glow-sweep {
  0% { transform: translateX(-25%); }
  100% { transform: translateX(25%); }
}

.ad-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  position: relative;
  z-index: 2;
}

.ad-icon {
  font-size: 28px;
  animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

.ad-main {
  font-size: 22px;
  font-weight: 700;
  background: linear-gradient(
    90deg, 
    #f0b429 0%, 
    #ff6b6b 25%, 
    #4ecdc4 50%, 
    #45b7d1 75%, 
    #f0b429 100%
  );
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradient-flow 3s linear infinite;
}

@keyframes gradient-flow {
  0% { background-position: 0% center; }
  100% { background-position: 200% center; }
}

.ad-sub {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.5);
  animation: fade-pulse 2s ease-in-out infinite;
}

@keyframes fade-pulse {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 0.8; }
}

.ad-particles {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  overflow: hidden;
  pointer-events: none;
}

.particle {
  position: absolute;
  width: 4px;
  height: 4px;
  background: var(--primary);
  border-radius: 50%;
  opacity: 0;
  animation: particle-float 4s ease-in-out infinite;
  animation-delay: calc(var(--i) * 0.3s);
}

.particle:nth-child(odd) {
  background: #4ecdc4;
}

.particle:nth-child(3n) {
  background: #ff6b6b;
}

.particle:nth-child(1) { left: 5%; top: 80%; }
.particle:nth-child(2) { left: 15%; top: 60%; }
.particle:nth-child(3) { left: 25%; top: 90%; }
.particle:nth-child(4) { left: 35%; top: 70%; }
.particle:nth-child(5) { left: 45%; top: 85%; }
.particle:nth-child(6) { left: 55%; top: 65%; }
.particle:nth-child(7) { left: 65%; top: 95%; }
.particle:nth-child(8) { left: 75%; top: 75%; }
.particle:nth-child(9) { left: 85%; top: 60%; }
.particle:nth-child(10) { left: 95%; top: 80%; }
.particle:nth-child(11) { left: 10%; top: 50%; }
.particle:nth-child(12) { left: 90%; top: 55%; }

@keyframes particle-float {
  0% {
    opacity: 0;
    transform: translateY(0) scale(0);
  }
  20% {
    opacity: 0.8;
    transform: translateY(-10px) scale(1);
  }
  80% {
    opacity: 0.4;
    transform: translateY(-40px) scale(0.6);
  }
  100% {
    opacity: 0;
    transform: translateY(-60px) scale(0);
  }
}

.ad-banner:hover .ad-glow {
  animation-duration: 1.5s;
}

.ad-banner:hover .ad-main {
  animation-duration: 1.5s;
}

.ad-banner:hover .particle {
  animation-duration: 2s;
}

/* 底部信息 */
.footer-info {
  margin-top: 20px;
  padding-top: 0;
}

.tech-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.tech-tag {
  padding: 6px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  font-size: 12px;
  color: var(--text-secondary);
}

.author {
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
}

.author strong {
  color: var(--primary);
}

/* 右侧面板 */
.right-panel {
  width: 360px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex-shrink: 0;
}

.status-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.card-header .el-icon {
  color: var(--primary);
}

.card-header .el-tag {
  margin-left: auto;
}

.status-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--bg-darker);
  border-radius: 8px;
}

.status-row .label {
  color: var(--text-muted);
  font-size: 13px;
}

.status-row .value {
  color: var(--text-primary);
  font-weight: 500;
  font-size: 13px;
  font-family: var(--font-mono);
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 30px;
  color: var(--text-muted);
}

/* 模型状态 */
.model-card {
  flex: 1;
}

.model-selector {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.model-type-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 16px;
  background: var(--bg-darker);
  border: 2px solid transparent;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.model-type-btn:hover {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.05);
}

.model-type-btn.active {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.1);
}

.model-type-btn .model-icon {
  font-size: 18px;
}

.model-type-btn .model-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.model-path-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  margin-bottom: 12px;
  background: var(--bg-lighter);
  border-radius: 6px;
  font-size: 12px;
}

.model-path-info .path-label {
  color: var(--text-muted);
  flex-shrink: 0;
}

.model-path-info .path-value {
  color: var(--text-primary);
  font-family: 'Consolas', 'Monaco', monospace;
  background: transparent;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-unchecked {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 24px;
  color: var(--text-muted);
  font-size: 13px;
}

.model-unchecked .el-icon {
  font-size: 18px;
  color: var(--warning);
}

.model-status {
  display: flex;
  align-items: center;
  gap: 24px;
  margin-bottom: 16px;
}

.model-ring {
  position: relative;
  width: 90px;
  height: 90px;
  flex-shrink: 0;
}

.model-ring svg {
  transform: rotate(-90deg);
  width: 100%;
  height: 100%;
}

.model-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.ring-bg {
  stroke: var(--bg-darker);
}

.ring-progress {
  stroke: var(--success);
  stroke-dasharray: 264;
  transition: stroke-dashoffset 0.5s ease;
}

.ring-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.ring-num {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
}

.ring-label {
  font-size: 12px;
  color: var(--text-muted);
}

.model-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.detail-row span {
  font-size: 12px;
  color: var(--text-muted);
}

.detail-row strong {
  font-size: 20px;
  color: var(--text-primary);
}

.detail-row strong.success {
  color: var(--success);
}

.component-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.comp-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg-darker);
  border-radius: 6px;
  font-size: 12px;
  color: var(--text-muted);
}

.comp-item.valid {
  background: rgba(103, 194, 58, 0.1);
  color: var(--success);
}

.comp-item.missing {
  opacity: 0.5;
}

.download-btn {
  width: 100%;
}

.download-progress-box {
  padding: 12px;
  background: var(--bg-lighter);
  border-radius: 8px;
}

.download-progress-box .progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 13px;
}

.download-progress-box .progress-percent {
  font-weight: 600;
  color: var(--success);
}

.download-progress-box .progress-info {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-muted);
}

/* 联系方式 */
.contact-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.contact-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.contact-row:hover {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.03);
}

.contact-row:hover .copy-icon {
  color: var(--primary);
}

.contact-icon {
  font-size: 18px;
}

.contact-text {
  flex: 1;
  font-size: 13px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
}

.copy-icon {
  color: var(--text-muted);
  font-size: 14px;
  transition: color 0.2s;
}

/* 响应式 */
@media (max-width: 1000px) {
  .welcome-page {
    flex-direction: column;
    overflow-y: auto;
    padding: 20px;
    gap: 20px;
  }
  
  .right-panel {
    width: 100%;
  }
  
  .nav-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 600px) {
  .nav-grid {
    grid-template-columns: 1fr;
  }
  
  .brand {
    flex-direction: column;
    text-align: center;
  }
}
</style>



