<template>
  <div class="model-management">
    <div class="page-header">
      <h2>模型管理</h2>
      <p class="page-description">使用抽象层进行模型检测、完整性验证和统一管理</p>
    </div>
    
    <ModelDetectionPanel />
  </div>
</template>

<script setup lang="ts">
import ModelDetectionPanel from '@/components/ModelDetectionPanel.vue'
</script>

<style scoped>
.model-management {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0 0 8px 0;
  color: #303133;
  font-size: 24px;
  font-weight: 600;
}

.page-description {
  margin: 0;
  color: #909399;
  font-size: 14px;
}
</style>