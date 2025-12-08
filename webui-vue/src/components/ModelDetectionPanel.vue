<template>
  <div class="model-detection-panel">
    <div class="panel-header">
      <h3>模型检测与管理</h3>
      <el-button-group>
        <el-button size="small" @click="refreshModelStatus" :loading="loading">
          <el-icon><Refresh /></el-icon> 刷新
        </el-button>
        <el-button size="small" type="primary" @click="showDownloadDialog = true">
          <el-icon><Download /></el-icon> 下载模型
        </el-button>
      </el-button-group>
    </div>

    <!-- 支持的模型列表 -->
    <div class="supported-models" v-if="supportedModels.length > 0">
      <h4>支持的模型类型</h4>
      <el-row :gutter="16">
        <el-col :span="12" v-for="model in supportedModels" :key="model.type">
          <el-card class="model-card" shadow="hover">
            <div class="model-info">
              <div class="model-header">
                <h5>{{ model.name }}</h5>
                <el-tag :type="getModelStatusType(model.status)" size="small">
                  {{ getModelStatusText(model.status) }}
                </el-tag>
              </div>
              <p class="model-description">{{ model.description }}</p>
              <div class="model-meta">
                <span class="model-size">{{ model.size_gb }} GB</span>
                <span class="model-id">{{ model.model_id }}</span>
              </div>
              <div class="model-actions">
                <el-button 
                  size="small" 
                  @click="detectModel(model.type)"
                  :loading="model.loading"
                >
                  检测状态
                </el-button>
                <el-button 
                  v-if="model.status !== 'valid'"
                  size="small" 
                  type="primary"
                  @click="startDownload(model.type)"
                  :loading="model.downloading"
                >
                  下载
                </el-button>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 模型检测结果 -->
    <div class="detection-results" v-if="detectionResults.length > 0">
      <h4>检测结果</h4>
      <el-collapse v-model="activeResults">
        <el-collapse-item 
          v-for="result in detectionResults" 
          :key="result.model_type"
          :title="`${result.model_name} - ${getStatusText(result.status)}`"
          :name="result.model_type"
        >
          <div class="result-details">
            <div class="result-summary">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="模型类型">{{ result.model_type }}</el-descriptions-item>
                <el-descriptions-item label="状态">
                  <el-tag :type="getStatusType(result.status)">
                    {{ getStatusText(result.status) }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="路径">{{ result.path }}</el-descriptions-item>
                <el-descriptions-item label="组件统计">
                  {{ result.summary.valid_components }}/{{ result.summary.total_components }}
                </el-descriptions-item>
              </el-descriptions>
            </div>

            <div class="component-details" v-if="result.components">
              <h5>组件详情</h5>
              <el-table :data="formatComponents(result.components)" style="width: 100%">
                <el-table-column prop="name" label="组件名称" width="150" />
                <el-table-column prop="required" label="必需" width="80">
                  <template #default="scope">
                    <el-tag :type="scope.row.required ? 'danger' : 'info'" size="small">
                      {{ scope.row.required ? '是' : '否' }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="status" label="状态" width="100">
                  <template #default="scope">
                    <el-tag :type="getComponentStatusType(scope.row.status)" size="small">
                      {{ getComponentStatusText(scope.row.status) }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="message" label="消息" />
              </el-table>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>

    <!-- 下载进度 -->
    <div class="download-progress" v-if="downloadTasks.length > 0">
      <h4>下载任务</h4>
      <el-card v-for="task in downloadTasks" :key="task.model_type" class="download-task">
        <div class="task-header">
          <span class="task-title">{{ task.model_name }}</span>
          <el-tag :type="getDownloadStatusType(task.status)" size="small">
            {{ getDownloadStatusText(task.status) }}
          </el-tag>
        </div>
        <div class="task-progress">
          <el-progress 
            :percentage="task.progress_percent" 
            :status="getProgressStatus(task.status)"
          />
          <div class="task-stats">
            <span>{{ formatSize(task.downloaded_mb) }} / {{ formatSize(task.total_mb) }}</span>
            <span v-if="task.speed_mbps > 0">{{ task.speed_mbps.toFixed(2) }} MB/s</span>
            <span v-if="task.eta_seconds">{{ formatETA(task.eta_seconds) }}</span>
          </div>
          <div class="current-file" v-if="task.current_file">
            当前文件: {{ task.current_file }}
          </div>
        </div>
        <div class="task-actions">
          <el-button 
            v-if="task.status === 'downloading'"
            size="small" 
            @click="pauseDownload(task.model_type)"
          >
            暂停
          </el-button>
          <el-button 
            v-if="task.status === 'paused'"
            size="small" 
            type="primary"
            @click="resumeDownload(task.model_type)"
          >
            继续
          </el-button>
          <el-button 
            v-if="['downloading', 'paused'].includes(task.status)"
            size="small" 
            type="danger"
            @click="cancelDownload(task.model_type)"
          >
            取消
          </el-button>
        </div>
      </el-card>
    </div>

    <!-- 下载对话框 -->
    <el-dialog
      v-model="showDownloadDialog"
      title="模型下载"
      width="600px"
    >
      <div class="download-dialog-content">
        <el-form :model="downloadForm" label-width="100px">
          <el-form-item label="选择模型">
            <el-select v-model="downloadForm.modelType" placeholder="请选择模型">
              <el-option
                v-for="model in downloadableModels"
                :key="model.type"
                :label="`${model.name} (${model.size_gb}GB)`"
                :value="model.type"
              >
                <span style="float: left">{{ model.name }}</span>
                <span style="float: right; color: #8492a6; font-size: 13px">
                  {{ model.size_gb }} GB
                </span>
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="保存路径">
            <el-input v-model="downloadForm.path" placeholder="留空使用默认路径">
              <template #append>
                <el-button @click="selectDownloadPath">选择</el-button>
              </template>
            </el-input>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showDownloadDialog = false">取消</el-button>
          <el-button type="primary" @click="confirmDownload" :loading="downloadLoading">
            开始下载
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, Download } from '@element-plus/icons-vue'

// 响应式数据
const loading = ref(false)
const showDownloadDialog = ref(false)
const downloadLoading = ref(false)
const activeResults = ref([])

const supportedModels = ref([])
const detectionResults = ref([])
const downloadTasks = ref([])

const downloadForm = reactive({
  modelType: '',
  path: ''
})

// 轮询定时器
let progressTimer = null

// 计算属性
const downloadableModels = computed(() => {
  return supportedModels.value.filter(model => model.supports_download)
})

// 方法
const refreshModelStatus = async () => {
  loading.value = true
  try {
    // 获取支持的模型列表
    const response = await fetch('/api/system/model-detector/supported-models')
    const result = await response.json()
    
    if (result.success) {
      supportedModels.value = Object.entries(result.models).map(([type, info]) => ({
        type,
        name: info.name,
        description: info.description,
        size_gb: info.size_gb,
        model_id: info.model_id,
        supports_detection: info.supports_detection,
        supports_download: info.supports_download,
        status: 'unknown',
        loading: false,
        downloading: false
      }))
      
      // 检测每个模型的状态
      for (const model of supportedModels.value) {
        if (model.supports_detection) {
          await detectModel(model.type, false)
        }
      }
    } else {
      ElMessage.error('获取模型列表失败: ' + result.error)
    }
  } catch (error) {
    ElMessage.error('刷新失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const detectModel = async (modelType, showMessage = true) => {
  const model = supportedModels.value.find(m => m.type === modelType)
  if (!model) return
  
  model.loading = true
  try {
    const response = await fetch(`/api/system/model-detector/detect/${modelType}`)
    const result = await response.json()
    
    if (result.success) {
      // 更新模型状态
      model.status = result.status
      
      // 添加到检测结果
      const existingIndex = detectionResults.value.findIndex(r => r.model_type === modelType)
      if (existingIndex >= 0) {
        detectionResults.value[existingIndex] = result
      } else {
        detectionResults.value.push(result)
      }
      
      if (showMessage) {
        ElMessage.success(`检测完成: ${result.model_name}`)
      }
    } else {
      if (showMessage) {
        ElMessage.error('检测失败: ' + result.error)
      }
    }
  } catch (error) {
    if (showMessage) {
      ElMessage.error('检测出错: ' + error.message)
    }
  } finally {
    model.loading = false
  }
}

const startDownload = async (modelType) => {
  const model = supportedModels.value.find(m => m.type === modelType)
  if (!model) return
  
  model.downloading = true
  try {
    const response = await fetch(`/api/system/model-downloader/start/${modelType}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({})
    })
    const result = await response.json()
    
    if (result.success) {
      ElMessage.success('下载任务已启动')
      
      // 添加到下载任务列表
      const existingIndex = downloadTasks.value.findIndex(t => t.model_type === modelType)
      if (existingIndex >= 0) {
        downloadTasks.value[existingIndex] = {
          ...downloadTasks.value[existingIndex],
          status: 'downloading',
          progress_percent: result.progress_percent
        }
      } else {
        downloadTasks.value.push({
          model_type: modelType,
          model_name: model.name,
          status: 'downloading',
          progress_percent: result.progress_percent,
          downloaded_mb: 0,
          total_mb: model.size_gb * 1024,
          speed_mbps: 0,
          eta_seconds: null,
          current_file: null
        })
      }
      
      // 开始轮询进度
      startProgressPolling()
    } else {
      ElMessage.error('启动下载失败: ' + result.error)
    }
  } catch (error) {
    ElMessage.error('下载出错: ' + error.message)
  } finally {
    model.downloading = false
  }
}

const confirmDownload = async () => {
  if (!downloadForm.modelType) {
    ElMessage.warning('请选择要下载的模型')
    return
  }
  
  downloadLoading.value = true
  try {
    const response = await fetch(`/api/system/model-downloader/start/${downloadForm.modelType}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        path: downloadForm.path || undefined
      })
    })
    const result = await response.json()
    
    if (result.success) {
      ElMessage.success('下载任务已启动')
      showDownloadDialog.value = false
      
      // 刷新状态
      refreshModelStatus()
      
      // 开始轮询进度
      startProgressPolling()
    } else {
      ElMessage.error('启动下载失败: ' + result.error)
    }
  } catch (error) {
    ElMessage.error('下载出错: ' + error.message)
  } finally {
    downloadLoading.value = false
  }
}

const selectDownloadPath = () => {
  // 这里可以实现路径选择功能
  ElMessage.info('路径选择功能待实现')
}

const getDownloadProgress = async (modelType) => {
  try {
    const response = await fetch(`/api/system/model-downloader/progress/${modelType}`)
    const result = await response.json()
    
    if (result.success) {
      // 更新下载任务状态
      const taskIndex = downloadTasks.value.findIndex(t => t.model_type === modelType)
      if (taskIndex >= 0) {
        downloadTasks.value[taskIndex] = {
          ...downloadTasks.value[taskIndex],
          status: result.status,
          progress_percent: result.progress_percent,
          downloaded_mb: result.downloaded_mb,
          total_mb: result.total_mb,
          speed_mbps: result.speed_mbps,
          eta_seconds: result.eta_seconds,
          current_file: result.current_file
        }
      }
      
      // 如果下载完成，刷新模型状态
      if (result.status === 'completed') {
        ElMessage.success('模型下载完成')
        await detectModel(modelType, false)
      } else if (result.status === 'failed') {
        ElMessage.error('下载失败: ' + result.error_message)
      }
    }
  } catch (error) {
    console.error('获取进度失败:', error)
  }
}

const cancelDownload = async (modelType) => {
  try {
    const response = await fetch(`/api/system/model-downloader/cancel/${modelType}`, {
      method: 'POST'
    })
    const result = await response.json()
    
    if (result.success) {
      ElMessage.success('下载已取消')
      // 从任务列表中移除
      downloadTasks.value = downloadTasks.value.filter(t => t.model_type !== modelType)
      
      // 如果没有任务了，停止轮询
      if (downloadTasks.value.length === 0) {
        stopProgressPolling()
      }
    } else {
      ElMessage.error('取消失败: ' + result.error)
    }
  } catch (error) {
    ElMessage.error('操作失败: ' + error.message)
  }
}

const startProgressPolling = () => {
  if (progressTimer) return
  
  progressTimer = setInterval(() => {
    downloadTasks.value.forEach(task => {
      if (['downloading', 'paused'].includes(task.status)) {
        getDownloadProgress(task.model_type)
      }
    })
  }, 2000) // 每2秒更新一次
}

const stopProgressPolling = () => {
  if (progressTimer) {
    clearInterval(progressTimer)
    progressTimer = null
  }
}

// 辅助函数
const formatComponents = (components) => {
  return Object.entries(components).map(([name, info]) => ({
    name,
    required: info.required,
    status: info.status,
    message: info.message
  }))
}

const getModelStatusType = (status) => {
  const typeMap = {
    'valid': 'success',
    'incomplete': 'warning',
    'corrupted': 'danger',
    'missing': 'info',
    'unknown': 'info'
  }
  return typeMap[status] || 'info'
}

const getModelStatusText = (status) => {
  const textMap = {
    'valid': '完整',
    'incomplete': '不完整',
    'corrupted': '损坏',
    'missing': '缺失',
    'unknown': '未知'
  }
  return textMap[status] || '未知'
}

const getStatusType = (status) => {
  const typeMap = {
    'valid': 'success',
    'incomplete': 'warning', 
    'corrupted': 'danger',
    'missing': 'info'
  }
  return typeMap[status] || 'info'
}

const getStatusText = (status) => {
  const textMap = {
    'valid': '有效',
    'incomplete': '不完整',
    'corrupted': '损坏',
    'missing': '缺失'
  }
  return textMap[status] || status
}

const getComponentStatusType = (status) => {
  return getStatusType(status)
}

const getComponentStatusText = (status) => {
  return getStatusText(status)
}

const getDownloadStatusType = (status) => {
  const typeMap = {
    'pending': 'info',
    'downloading': 'primary',
    'paused': 'warning',
    'completed': 'success',
    'failed': 'danger',
    'cancelled': 'info'
  }
  return typeMap[status] || 'info'
}

const getDownloadStatusText = (status) => {
  const textMap = {
    'pending': '等待中',
    'downloading': '下载中',
    'paused': '已暂停',
    'completed': '已完成',
    'failed': '失败',
    'cancelled': '已取消'
  }
  return textMap[status] || status
}

const getProgressStatus = (status) => {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'exception'
  return null
}

const formatSize = (mb) => {
  if (mb >= 1024) {
    return (mb / 1024).toFixed(2) + ' GB'
  }
  return mb.toFixed(2) + ' MB'
}

const formatETA = (seconds) => {
  if (seconds < 60) {
    return Math.ceil(seconds) + ' 秒'
  } else if (seconds < 3600) {
    return Math.ceil(seconds / 60) + ' 分钟'
  } else {
    return (seconds / 3600).toFixed(1) + ' 小时'
  }
}

// 生命周期
onMounted(() => {
  refreshModelStatus()
})

onUnmounted(() => {
  stopProgressPolling()
})
</script>

<style scoped>
.model-detection-panel {
  padding: 20px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.panel-header h3 {
  margin: 0;
}

.supported-models {
  margin-bottom: 30px;
}

.supported-models h4 {
  margin-bottom: 16px;
}

.model-card {
  margin-bottom: 16px;
}

.model-info {
  padding: 10px 0;
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.model-header h5 {
  margin: 0;
  font-size: 16px;
}

.model-description {
  color: #606266;
  font-size: 14px;
  margin: 10px 0;
  line-height: 1.5;
}

.model-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 10px 0;
  font-size: 12px;
  color: #909399;
}

.model-actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}

.detection-results {
  margin-bottom: 30px;
}

.detection-results h4 {
  margin-bottom: 16px;
}

.result-details {
  padding: 10px;
}

.result-summary {
  margin-bottom: 20px;
}

.component-details h5 {
  margin: 20px 0 10px 0;
}

.download-progress h4 {
  margin-bottom: 16px;
}

.download-task {
  margin-bottom: 16px;
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.task-title {
  font-weight: bold;
  font-size: 16px;
}

.task-progress {
  margin-bottom: 15px;
}

.task-stats {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 12px;
  color: #606266;
}

.current-file {
  margin-top: 10px;
  font-size: 12px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-actions {
  display: flex;
  gap: 8px;
}

.download-dialog-content {
  padding: 20px 0;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>