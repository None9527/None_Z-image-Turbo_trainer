<template>
  <div 
    class="paired-card glass-card"
    :class="{ selected: isSelected }"
    @click="$emit('click', pair)"
  >
    <!-- 选择圆圈 -->
    <div 
      class="select-circle"
      :class="{ checked: isSelected }"
      @click.stop="$emit('toggle-select', pair.target.path)"
    >
      <el-icon v-if="isSelected"><Check /></el-icon>
    </div>
    
    <!-- 缓存状态标签 -->
    <div class="cache-tags">
      <div class="cache-tag" :class="{ active: pair.target.hasLatentCache }" title="Latent 缓存">
        <span>L</span>
      </div>
      <div class="cache-tag" :class="{ active: pair.target.hasTextCache }" title="Text 缓存">
        <span>T</span>
      </div>
    </div>
    
    <!-- 配对图片区域 -->
    <div class="pair-images">
      <!-- Source 图片 -->
      <div class="image-slot source-slot">
        <div class="slot-label">Source</div>
        <div class="image-wrapper" v-if="pair.source">
          <img 
            :src="pair.source.thumbnailUrl" 
            :alt="pair.source.filename"
            loading="lazy"
            @error="handleImageError($event, 'source')"
          />
        </div>
        <div class="image-placeholder" v-else>
          <el-icon><Picture /></el-icon>
          <span>无源图</span>
        </div>
      </div>
      
      <!-- 箭头指示 -->
      <div class="arrow-indicator">
        <el-icon><Right /></el-icon>
      </div>
      
      <!-- Target 图片 -->
      <div class="image-slot target-slot">
        <div class="slot-label">Target</div>
        <div class="image-wrapper">
          <img 
            :src="pair.target.thumbnailUrl" 
            :alt="pair.target.filename"
            loading="lazy"
            @error="handleImageError($event, 'target')"
          />
        </div>
      </div>
    </div>
    
    <!-- 图片信息 -->
    <div class="pair-info">
      <div class="pair-name" :title="pair.id">{{ pair.id }}</div>
      <div class="pair-meta">
        <span v-if="pair.target.width && pair.target.height">
          {{ pair.target.width }}×{{ pair.target.height }}
        </span>
        <span v-if="pair.source"> · 已配对</span>
        <span v-else class="unpaired"> · 缺少源图</span>
      </div>
      <div class="pair-caption" :class="{ 'no-caption': !pair.target.caption }">
        {{ pair.target.caption || '无标注' }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Check, Picture, Right } from '@element-plus/icons-vue'
import type { ImagePair } from '@/stores/dataset'

const props = defineProps<{
  pair: ImagePair
  selected?: boolean
}>()

defineEmits<{
  (e: 'click', pair: ImagePair): void
  (e: 'toggle-select', path: string): void
}>()

const isSelected = computed(() => props.selected || false)

function handleImageError(event: Event, slot: 'source' | 'target') {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
  console.warn(`[PairedCard] Image load failed: ${slot}`)
}
</script>

<style lang="scss" scoped>
.paired-card {
  padding: var(--space-sm);
  cursor: pointer;
  transition: all var(--transition-fast);
  position: relative;
  display: flex;
  flex-direction: column;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
  }
  
  &.selected {
    border-color: var(--primary);
    box-shadow: 0 0 20px var(--primary-glow);
  }
}

.select-circle {
  position: absolute;
  top: var(--space-sm);
  right: var(--space-sm);
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.6);
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
  z-index: 2;
  
  &:hover {
    border-color: var(--primary);
    background: rgba(0, 0, 0, 0.6);
  }
  
  &.checked {
    border-color: var(--primary);
    background: var(--primary);
    
    .el-icon {
      color: var(--bg-dark);
      font-size: 14px;
      font-weight: bold;
    }
  }
}

.cache-tags {
  position: absolute;
  top: var(--space-xs);
  left: var(--space-xs);
  display: flex;
  gap: 4px;
  z-index: 2;
}

.cache-tag {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: rgba(0, 0, 0, 0.7);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 600;
  backdrop-filter: blur(4px);
  
  &.active {
    background: var(--success);
    color: white;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
  }
}

.pair-images {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  padding: var(--space-sm);
  background: var(--bg-darker);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-sm);
}

.image-slot {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.slot-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.image-wrapper {
  width: 100%;
  aspect-ratio: 1;
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--bg-dark);
  
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform var(--transition-fast);
  }
  
  &:hover img {
    transform: scale(1.05);
  }
}

.image-placeholder {
  width: 100%;
  aspect-ratio: 1;
  border-radius: var(--radius-sm);
  background: var(--bg-dark);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  color: var(--text-muted);
  
  .el-icon {
    font-size: 24px;
    opacity: 0.5;
  }
  
  span {
    font-size: 10px;
  }
}

.arrow-indicator {
  color: var(--primary);
  font-size: 20px;
  opacity: 0.7;
  flex-shrink: 0;
}

.pair-info {
  padding: var(--space-xs) var(--space-sm);
  
  .pair-name {
    font-size: 0.85rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: var(--space-xs);
  }
  
  .pair-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: var(--space-xs);
    
    .unpaired {
      color: var(--warning);
    }
  }
  
  .pair-caption {
    font-size: 0.75rem;
    color: var(--text-secondary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    
    &.no-caption {
      color: var(--text-muted);
      font-style: italic;
    }
  }
}
</style>
