<template>
  <el-dialog
    v-model="visible"
    title="生成缓存"
    width="550px"
    @close="handleClose"
  >
    <el-form label-width="auto">
      <el-form-item label="模型类型">
        <el-select v-model="config.modelType" placeholder="请选择模型类型">
          <el-option label="Z-Image" value="zimage" />
        </el-select>
        <div class="cache-model-hint">
          <span>将使用 Z-Image 缓存脚本生成 _zi.safetensors 格式的缓存文件</span>
        </div>
      </el-form-item>
      
      <el-form-item label="训练模式">
        <el-select v-model="config.trainingMode" placeholder="请选择训练模式">
          <el-option label="Text2Img (标准)" value="text2img" />
          <el-option label="ControlNet" value="controlnet" />
          <el-option label="Img2Img" value="img2img" />
          <el-option label="Omni (多图)" value="omni" />
        </el-select>
        <div class="cache-model-hint">
          <span v-if="config.trainingMode === 'text2img'">标准 LoRA 训练，只需 latent 和 text 缓存</span>
          <span v-else-if="config.trainingMode === 'controlnet'">ControlNet 训练，需要额外的条件图缓存</span>
          <span v-else-if="config.trainingMode === 'img2img'">Img2Img 训练，需要源图和目标图 latent 配对</span>
          <span v-else-if="config.trainingMode === 'omni'">多图条件训练，需要 SigLIP 特征缓存</span>
        </div>
      </el-form-item>
      
      <el-form-item label="选择缓存类型">
        <el-checkbox-group v-model="config.options">
          <el-checkbox label="latent">
            Latent 缓存
            <span class="cache-path-hint" v-if="vaePath">({{ vaePath.split(/[/\\]/).pop() }})</span>
            <span class="cache-path-missing" v-else>(未配置VAE)</span>
          </el-checkbox>
          <el-checkbox label="text">
            Text 缓存
            <span class="cache-path-hint" v-if="textEncoderPath">({{ textEncoderPath.split(/[/\\]/).pop() }})</span>
            <span class="cache-path-missing" v-else>(未配置Text Encoder)</span>
          </el-checkbox>
          
          <!-- ControlNet 模式额外选项 -->
          <el-checkbox label="control" v-if="config.trainingMode === 'controlnet'">
            条件图缓存 (边缘/深度/姿态)
          </el-checkbox>
          
          <!-- Omni 模式额外选项 -->
          <el-checkbox label="siglip" v-if="config.trainingMode === 'omni'">
            SigLIP 特征缓存
            <span class="cache-path-hint" v-if="siglipPath">({{ siglipPath.split(/[/\\]/).pop() }})</span>
            <span class="cache-path-missing" v-else>(需配置SigLIP)</span>
          </el-checkbox>
        </el-checkbox-group>
      </el-form-item>
      
      <!-- ControlNet 条件图目录 -->
      <el-form-item label="条件图目录" v-if="config.trainingMode === 'controlnet' && config.options.includes('control')">
        <el-input v-model="config.controlDir" placeholder="条件图像目录路径" />
      </el-form-item>
      
      <!-- Img2Img 源图目录 -->
      <el-form-item label="源图目录" v-if="config.trainingMode === 'img2img'">
        <el-input v-model="config.sourceDir" placeholder="源图像目录路径" />
      </el-form-item>
    </el-form>
    
    <div class="cache-warning" v-if="!hasRequiredPaths">
      <el-icon><WarningFilled /></el-icon>
      <span>请先在「训练配置」页面设置模型路径</span>
      <el-button type="primary" link @click="$emit('goToConfig')">前往配置</el-button>
    </div>
    
    <div class="cache-hint" v-else>
      <el-icon><InfoFilled /></el-icon>
      <span>缓存文件将保存在数据集目录中</span>
    </div>
    
    <template #footer>
      <el-button @click="handleClose">取消</el-button>
      <el-button type="primary" @click="handleConfirm" :loading="generating" :disabled="!canGenerate">
        开始生成
      </el-button>
    </template>
  </el-dialog>
</template>


<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { InfoFilled, WarningFilled } from '@element-plus/icons-vue'

// Props
const props = defineProps<{
  modelValue: boolean
  vaePath?: string
  textEncoderPath?: string
  siglipPath?: string
  generating?: boolean
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', config: { 
    modelType: string
    trainingMode: string
    options: string[]
    controlDir?: string
    sourceDir?: string 
  }): void
  (e: 'goToConfig'): void
}>()

// 本地状态
const visible = ref(props.modelValue)
const config = ref({
  modelType: 'zimage',
  trainingMode: 'text2img',
  options: ['latent', 'text'] as string[],
  controlDir: '',
  sourceDir: '',
})


// 同步 visible 与 modelValue
watch(() => props.modelValue, (val) => {
  visible.value = val
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

// 是否配置了必要路径
const hasRequiredPaths = computed(() => {
  return (config.value.options.includes('latent') && props.vaePath) ||
         (config.value.options.includes('text') && props.textEncoderPath)
})

// 是否可以生成
const canGenerate = computed(() => {
  return config.value.options.length > 0 && hasRequiredPaths.value
})

// 确认生成
function handleConfirm() {
  emit('confirm', { ...config.value })
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.cache-model-hint {
  margin-top: var(--space-xs);
  font-size: 12px;
  color: var(--text-muted);
}

.cache-path-hint {
  font-size: 12px;
  color: var(--text-muted);
}

.cache-path-missing {
  font-size: 12px;
  color: var(--color-warning);
}

.cache-warning, .cache-hint {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  font-size: 13px;
}

.cache-warning {
  background: var(--color-warning-light);
  color: var(--color-warning-dark);
}

.cache-hint {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}
</style>
