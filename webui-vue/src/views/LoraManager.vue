<template>
  <div class="model-manager">
    <div class="page-header">
      <h1><el-icon><Files /></el-icon> 模型管理</h1>
      <el-button @click="fetchModels" :loading="loading">
        <el-icon><Refresh /></el-icon> 刷新
      </el-button>
    </div>

    <!-- 模型类型 Tabs -->
    <el-tabs v-model="activeTab" type="card" class="model-tabs" @tab-change="handleTabChange">
      <el-tab-pane label="LoRA" name="lora">
        <template #label>
          <span class="tab-label">
            <el-icon><Connection /></el-icon> LoRA
            <el-badge v-if="loraList.length > 0" :value="loraList.length" class="tab-badge" />
          </span>
        </template>
      </el-tab-pane>
      <el-tab-pane label="Finetune" name="finetune">
        <template #label>
          <span class="tab-label">
            <el-icon><Cpu /></el-icon> Finetune
            <el-badge v-if="finetuneList.length > 0" :value="finetuneList.length" class="tab-badge" />
          </span>
        </template>
      </el-tab-pane>
      <el-tab-pane label="ControlNet" name="controlnet">
        <template #label>
          <span class="tab-label">
            <el-icon><Grid /></el-icon> ControlNet
            <el-badge v-if="controlnetList.length > 0" :value="controlnetList.length" class="tab-badge" />
          </span>
        </template>
      </el-tab-pane>
    </el-tabs>

    <el-card class="model-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <span>{{ currentTabLabel }} ({{ currentList.length }} 个模型)</span>
            <!-- 批量操作按钮 -->
            <div class="batch-actions" v-if="selectedModels.length > 0">
              <el-button type="primary" size="small" @click="batchDownload">
                <el-icon><Download /></el-icon> 下载选中 ({{ selectedModels.length }})
              </el-button>
              <el-button type="danger" size="small" @click="batchDelete">
                <el-icon><Delete /></el-icon> 删除选中 ({{ selectedModels.length }})
              </el-button>
            </div>
          </div>
          <div class="path-hint">
            <span>路径: {{ currentPath }}</span>
            <el-button 
              type="primary" 
              link 
              size="small" 
              @click="copyPath"
              class="copy-btn"
            >
              <el-icon><CopyDocument /></el-icon>
            </el-button>
          </div>
        </div>
      </template>

      <div v-loading="loading" class="model-content">
        <el-empty v-if="currentList.length === 0 && !loading" :description="`暂无 ${currentTabLabel} 模型`">
          <template #image>
            <el-icon style="font-size: 64px; color: var(--el-text-color-secondary)"><FolderOpened /></el-icon>
          </template>
        </el-empty>

        <el-table 
          v-else 
          :data="currentList" 
          style="width: 100%" 
          stripe
          @selection-change="handleSelectionChange"
          ref="tableRef"
        >
          <el-table-column type="selection" width="50" />
          
          <el-table-column prop="name" label="文件名" min-width="300">
            <template #default="{ row }">
              <div class="file-name">
                <el-icon class="file-icon"><Document /></el-icon>
                <span>{{ row.name }}</span>
                <el-tag v-if="row.is_default" type="success" size="small">默认</el-tag>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column prop="size" label="大小" width="120" align="right">
            <template #default="{ row }">
              {{ formatSize(row.size) }}
            </template>
          </el-table-column>

          <el-table-column label="操作" width="200" align="center">
            <template #default="{ row }">
              <el-button-group v-if="!row.is_default">
                <el-button type="primary" size="small" @click="downloadModel(row)">
                  <el-icon><Download /></el-icon> 下载
                </el-button>
                <el-button type="danger" size="small" @click="deleteModel(row)">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </el-button-group>
              <span v-else class="default-hint">系统模型</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <!-- 删除确认对话框（单个） -->
    <el-dialog v-model="deleteDialogVisible" title="确认删除" width="400px">
      <p>确定要删除此模型吗？</p>
      <p class="delete-filename">{{ selectedModel?.name }}</p>
      <p class="warning-text">此操作不可恢复！</p>
      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmDelete" :loading="deleting">删除</el-button>
      </template>
    </el-dialog>

    <!-- 批量删除确认对话框 -->
    <el-dialog v-model="batchDeleteDialogVisible" title="批量删除确认" width="500px">
      <p>确定要删除以下 {{ selectedModels.length }} 个模型吗？</p>
      <div class="batch-delete-list">
        <div v-for="model in selectedModels" :key="model.path" class="delete-item">
          <el-icon><Document /></el-icon>
          <span>{{ model.name }}</span>
          <span class="delete-item-size">{{ formatSize(model.size) }}</span>
        </div>
      </div>
      <p class="warning-text">⚠️ 此操作不可恢复！</p>
      <template #footer>
        <el-button @click="batchDeleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmBatchDelete" :loading="deleting">
          删除全部 ({{ selectedModels.length }})
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Files, Refresh, Document, Download, Delete, FolderOpened, CopyDocument, Connection, Cpu, Grid } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

interface ModelItem {
  name: string
  path: string
  size: number
  is_default?: boolean
}

const loading = ref(false)
const activeTab = ref('lora')

// 三种模型列表
const loraList = ref<ModelItem[]>([])
const finetuneList = ref<ModelItem[]>([])
const controlnetList = ref<ModelItem[]>([])

// 路径
const loraPath = ref('')
const finetunePath = ref('')
const controlnetPath = ref('')

const deleteDialogVisible = ref(false)
const batchDeleteDialogVisible = ref(false)
const selectedModel = ref<ModelItem | null>(null)
const selectedModels = ref<ModelItem[]>([])
const deleting = ref(false)
const tableRef = ref()

// 计算属性
const currentList = computed(() => {
  switch (activeTab.value) {
    case 'lora': return loraList.value
    case 'finetune': return finetuneList.value.filter(m => !m.is_default)
    case 'controlnet': return controlnetList.value
    default: return []
  }
})

const currentPath = computed(() => {
  switch (activeTab.value) {
    case 'lora': return loraPath.value
    case 'finetune': return finetunePath.value
    case 'controlnet': return controlnetPath.value
    default: return ''
  }
})

const currentTabLabel = computed(() => {
  switch (activeTab.value) {
    case 'lora': return 'LoRA'
    case 'finetune': return 'Finetune'
    case 'controlnet': return 'ControlNet'
    default: return ''
  }
})

const formatSize = (bytes: number) => {
  if (!bytes || bytes === 0) return '-'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}

const copyPath = async () => {
  try {
    await navigator.clipboard.writeText(currentPath.value)
    ElMessage.success('路径已复制到剪贴板')
  } catch (e) {
    const textarea = document.createElement('textarea')
    textarea.value = currentPath.value
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
    ElMessage.success('路径已复制到剪贴板')
  }
}

const handleTabChange = () => {
  selectedModels.value = []
}

const fetchModels = async () => {
  loading.value = true
  selectedModels.value = []
  
  try {
    // 并行获取三种模型
    const [loraRes, finetuneRes] = await Promise.all([
      axios.get('/api/loras'),
      axios.get('/api/transformers'),
    ])
    
    loraList.value = loraRes.data.loras || []
    loraPath.value = loraRes.data.loraPath || './output/lora'
    
    finetuneList.value = finetuneRes.data.transformers || []
    finetunePath.value = finetuneRes.data.finetunePath || './output/finetune'
    
    // ControlNet 目前可能没有 API，先置空
    controlnetList.value = []
    controlnetPath.value = './output/controlnet'
    
  } catch (e) {
    console.error('Failed to fetch models:', e)
    ElMessage.error('获取模型列表失败')
  } finally {
    loading.value = false
  }
}

const handleSelectionChange = (selection: ModelItem[]) => {
  selectedModels.value = selection.filter(m => !m.is_default)
}

const downloadModel = (model: ModelItem) => {
  const apiPath = activeTab.value === 'lora' ? '/api/loras/download' : '/api/loras/download'
  const link = document.createElement('a')
  link.href = `${apiPath}?path=${encodeURIComponent(model.path)}`
  link.setAttribute('download', model.name.split('/').pop() || 'model.safetensors')
  document.body.appendChild(link)
  link.click()
  link.remove()
  ElMessage.info('已开始下载')
}

const batchDownload = () => {
  if (selectedModels.value.length === 0) return
  
  ElMessage.info(`开始下载 ${selectedModels.value.length} 个文件...`)
  
  selectedModels.value.forEach((model, index) => {
    setTimeout(() => {
      downloadModel(model)
    }, index * 500)
  })
}

const deleteModel = (model: ModelItem) => {
  selectedModel.value = model
  deleteDialogVisible.value = true
}

const batchDelete = () => {
  if (selectedModels.value.length === 0) return
  batchDeleteDialogVisible.value = true
}

const confirmDelete = async () => {
  if (!selectedModel.value) return
  
  deleting.value = true
  try {
    await axios.delete(`/api/loras/delete?path=${encodeURIComponent(selectedModel.value.path)}`)
    ElMessage.success('删除成功')
    deleteDialogVisible.value = false
    fetchModels()
  } catch (e) {
    console.error('Delete failed:', e)
    ElMessage.error('删除失败')
  } finally {
    deleting.value = false
  }
}

const confirmBatchDelete = async () => {
  if (selectedModels.value.length === 0) return
  
  deleting.value = true
  let successCount = 0
  let failCount = 0
  
  try {
    for (const model of selectedModels.value) {
      try {
        await axios.delete(`/api/loras/delete?path=${encodeURIComponent(model.path)}`)
        successCount++
      } catch (e) {
        console.error('Delete failed:', model.name, e)
        failCount++
      }
    }
    
    if (failCount === 0) {
      ElMessage.success(`成功删除 ${successCount} 个文件`)
    } else {
      ElMessage.warning(`删除完成: ${successCount} 成功, ${failCount} 失败`)
    }
    
    batchDeleteDialogVisible.value = false
    fetchModels()
  } catch (e) {
    console.error('Batch delete error:', e)
    ElMessage.error('批量删除出错')
  } finally {
    deleting.value = false
  }
}

onMounted(() => {
  fetchModels()
})
</script>

<style scoped>
.model-manager {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h1 {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 24px;
  margin: 0;
}

.model-tabs {
  margin-bottom: 16px;
}

.tab-label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tab-badge {
  margin-left: 4px;
}

:deep(.el-tabs__item) {
  font-size: 14px;
}

.model-card {
  background: var(--el-bg-color);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.batch-actions {
  display: flex;
  gap: 8px;
}

.path-hint {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
}

.copy-btn {
  padding: 2px 4px;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.copy-btn:hover {
  opacity: 1;
}

.model-content {
  min-height: 300px;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-icon {
  color: var(--el-color-primary);
}

.default-hint {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.delete-filename {
  font-family: monospace;
  background: var(--el-fill-color-light);
  padding: 8px 12px;
  border-radius: 4px;
  word-break: break-all;
}

.warning-text {
  color: var(--el-color-danger);
  font-size: 12px;
  margin-top: 12px;
}

.batch-delete-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-light);
  border-radius: 4px;
  margin: 12px 0;
}

.delete-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  font-size: 13px;
}

.delete-item:last-child {
  border-bottom: none;
}

.delete-item .el-icon {
  color: var(--el-color-primary);
  flex-shrink: 0;
}

.delete-item span {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delete-item-size {
  flex: none !important;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

:deep(.el-table) {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
}

:deep(.el-table .el-table__header-wrapper th) {
  background: var(--el-fill-color-light);
}
</style>
