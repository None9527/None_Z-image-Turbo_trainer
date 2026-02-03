<template>
  <div 
    class="omni-card glass-card"
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
      <div class="cache-tag" :class="{ active: hasSiglipCache }" title="SigLIP 缓存">
        <span>S</span>
      </div>
    </div>
    
    <!-- 多图网格区域 -->
    <div class="omni-images">
      <!-- 条件图 (最多显示3个) -->
      <div 
        class="condition-slot" 
        v-for="(cond, idx) in displayConditions" 
        :key="idx"
      >
        <div class="slot-label">C{{ idx + 1 }}</div>
        <div class="image-wrapper">
          <img 
            :src="cond.thumbnailUrl" 
            :alt="cond.filename"
            loading="lazy"
          />
        </div>
      </div>
      
      <!-- 更多条件图指示器 -->
      <div class="more-indicator" v-if="hasMoreConditions">
        <span>+{{ moreConditionsCount }}</span>
      </div>
      
      <!-- Target 图片 (较大) -->
      <div class="target-slot">
        <div class="slot-label">Target</div>
        <div class="image-wrapper target-wrapper">
          <img 
            :src="pair.target.thumbnailUrl" 
            :alt="pair.target.filename"
            loading="lazy"
          />
        </div>
      </div>
    </div>
    
    <!-- 图片信息 -->
    <div class="omni-info">
      <div class="omni-name" :title="pair.id">{{ pair.id }}</div>
      <div class="omni-meta">
        <span v-if="pair.target.width && pair.target.height">
          {{ pair.target.width }}×{{ pair.target.height }}
        </span>
        <span class="condition-count">
          · 条件图: {{ conditionCount }}
        </span>
      </div>
      <div class="omni-caption" :class="{ 'no-caption': !pair.target.caption }">
        {{ pair.target.caption || '无标注' }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Check } from '@element-plus/icons-vue'
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

const conditionCount = computed(() => props.pair.conditions?.length || 0)

// 最多显示3个条件图
const maxDisplayConditions = 3
const displayConditions = computed(() => 
  props.pair.conditions?.slice(0, maxDisplayConditions) || []
)

const hasMoreConditions = computed(() => conditionCount.value > maxDisplayConditions)
const moreConditionsCount = computed(() => conditionCount.value - maxDisplayConditions)

// 检查是否有 SigLIP 缓存 (检查任一条件图)
const hasSiglipCache = computed(() => 
  props.pair.conditions?.some(c => c.hasSiglipCache) || 
  (props.pair.target as any).hasSiglipCache || 
  false
)
</script>

<style lang="scss" scoped>
.omni-card {
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

.omni-images {
  display: flex;
  align-items: flex-end;
  gap: var(--space-xs);
  padding: var(--space-sm);
  background: var(--bg-darker);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-sm);
}

.condition-slot {
  flex: 0 0 auto;
  width: 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.target-slot {
  flex: 1;
  min-width: 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.slot-label {
  font-size: 9px;
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

.target-wrapper {
  border: 2px solid var(--primary);
  box-shadow: 0 0 8px var(--primary-glow);
}

.more-indicator {
  flex: 0 0 auto;
  width: 32px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-dark);
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.omni-info {
  padding: var(--space-xs) var(--space-sm);
  
  .omni-name {
    font-size: 0.85rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: var(--space-xs);
  }
  
  .omni-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: var(--space-xs);
    
    .condition-count {
      color: var(--info);
    }
  }
  
  .omni-caption {
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
