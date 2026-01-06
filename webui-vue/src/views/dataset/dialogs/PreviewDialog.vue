<template>
  <el-dialog
    v-model="visible"
    title="图片预览与编辑"
    width="1200px"
    class="preview-edit-dialog"
    :close-on-click-modal="true"
    align-center
    @close="handleClose"
  >
    <div class="preview-edit-layout" v-if="image">
      <!-- 左侧：图片预览 -->
      <div class="preview-side">
        <div class="image-wrapper">
          <img :src="`/api/dataset/image?path=${encodeURIComponent(image.path)}`" :alt="image.filename" />
        </div>
        <div class="image-meta-info">
          <span v-if="image.width && image.height">{{ image.width }} x {{ image.height }}</span>
          <span v-else>尺寸未知</span>
          <span>{{ formatSize(image.size) }}</span>
          <span>{{ image.filename }}</span>
        </div>
      </div>
      
      <!-- 右侧：标注编辑 -->
      <div class="edit-side">
        <div class="edit-header">
          <h3>图片标注</h3>
          <div class="edit-actions">
            <el-button type="primary" @click="handleSave" :loading="saving">
              保存标注
            </el-button>
          </div>
        </div>
        
        <el-input
          v-model="localCaption"
          type="textarea"
          :rows="20"
          placeholder="输入图片描述..."
          resize="none"
          class="caption-textarea"
          @keydown.ctrl.enter="handleSave"
        />
        
        <div class="keyboard-hint">
          <el-icon><InfoFilled /></el-icon>
          <span>提示: 支持 Ctrl+Enter 快速保存</span>
        </div>
      </div>
    </div>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { InfoFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useDatasetStore, type DatasetImage } from '@/stores/dataset'

// Props
const props = defineProps<{
  modelValue: boolean
  image: DatasetImage | null
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'saved'): void
}>()

const datasetStore = useDatasetStore()

// 本地状态
const visible = ref(props.modelValue)
const localCaption = ref('')
const saving = ref(false)

// 同步 visible 与 modelValue
watch(() => props.modelValue, (val) => {
  visible.value = val
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

// 当 image 变化时更新 caption
watch(() => props.image, (img) => {
  localCaption.value = img?.caption || ''
}, { immediate: true })

// 格式化文件大小
function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

// 保存标注
async function handleSave() {
  if (!props.image) return
  
  saving.value = true
  try {
    const success = await datasetStore.saveCaption(props.image.path, localCaption.value)
    if (success) {
      ElMessage.success('标注已保存')
      emit('saved')
    } else {
      ElMessage.error('保存失败')
    }
  } finally {
    saving.value = false
  }
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.preview-edit-layout {
  display: flex;
  gap: var(--space-lg);
  height: 70vh;
  max-height: 800px;
}

.preview-side {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  
  .image-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border-radius: var(--radius-md);
    overflow: hidden;
    
    img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
  }
  
  .image-meta-info {
    display: flex;
    gap: var(--space-md);
    padding: var(--space-sm) 0;
    font-size: 12px;
    color: var(--text-muted);
  }
}

.edit-side {
  width: 400px;
  display: flex;
  flex-direction: column;
  
  .edit-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-md);
    
    h3 {
      margin: 0;
      font-size: 16px;
    }
  }
  
  .caption-textarea {
    flex: 1;
    
    :deep(.el-textarea__inner) {
      height: 100%;
      resize: none;
    }
  }
  
  .keyboard-hint {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    margin-top: var(--space-sm);
    font-size: 12px;
    color: var(--text-muted);
  }
}
</style>
