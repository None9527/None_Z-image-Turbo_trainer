<template>
  <el-config-provider :locale="zhCn">
    <div class="app-container" :class="{ 'dark': isDark }">
      <!-- 背景动画 -->
      <div class="bg-animation">
        <div class="grid-overlay"></div>
        <div class="glow-orb orb-1"></div>
        <div class="glow-orb orb-2"></div>
        <div class="glow-orb orb-3"></div>
      </div>

      <!-- 主布局 -->
      <el-container class="main-layout">
        <!-- 侧边栏 -->
        <el-aside width="240px" class="sidebar">
          <div class="logo">
            <div class="logo-icon">N</div>
            <div class="logo-text">
              <span class="title">NONE</span>
              <span class="subtitle">TRAINER</span>
            </div>
          </div>

          <el-menu
            :default-active="activeMenu"
            class="sidebar-menu"
            router
            :collapse="isCollapse"
          >
            <el-menu-item index="/welcome">
              <el-icon><HomeFilled /></el-icon>
              <span>首页</span>
            </el-menu-item>
            <el-menu-item index="/dataset">
              <el-icon><Picture /></el-icon>
              <span>数据集</span>
            </el-menu-item>
            <el-menu-item index="/config">
              <el-icon><Setting /></el-icon>
              <span>训练配置</span>
            </el-menu-item>
            <el-menu-item index="/training">
              <el-icon><VideoPlay /></el-icon>
              <span>开始训练</span>
            </el-menu-item>
            <el-menu-item index="/generation">
              <el-icon><MagicStick /></el-icon>
              <span>图片生成</span>
            </el-menu-item>
            <el-menu-item index="/monitor">
              <el-icon><DataLine /></el-icon>
              <span>训练监控</span>
            </el-menu-item>
            <el-menu-item index="/loras">
              <el-icon><Files /></el-icon>
              <span>模型管理</span>
            </el-menu-item>
          </el-menu>

          <!-- 系统状态 -->
          <div class="system-status">
            <div class="status-item">
              <el-icon><Cpu /></el-icon>
              <span>{{ gpuInfo.name || 'GPU' }}</span>
            </div>
            <div class="status-item">
              <el-progress 
                :percentage="gpuInfo.memoryPercent || 0" 
                :stroke-width="6"
                :show-text="false"
              />
              <span class="memory-text">{{ gpuInfo.memoryUsed || '0' }} / {{ gpuInfo.memoryTotal || '0' }} GB</span>
            </div>
          </div>
        </el-aside>

        <!-- 主内容区 -->
        <el-main class="main-content">
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </div>
  </el-config-provider>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useDark } from '@vueuse/core'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { HomeFilled, MagicStick, Files } from '@element-plus/icons-vue'

const route = useRoute()
const isDark = useDark()
const isCollapse = ref(false)
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

const activeMenu = computed(() => route.path)
const gpuInfo = computed(() => systemStore.gpuInfo)

// WebSocket 连接状态
const isWsConnected = computed(() => wsStore.isConnected)

onMounted(() => {
  // 使用 WebSocket 实时获取数据
  wsStore.connect()
  wsStore.startHeartbeat()
})

onUnmounted(() => {
  wsStore.stopHeartbeat()
  wsStore.disconnect()
})
</script>

<style lang="scss" scoped>
.app-container {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-dark);
  color: var(--text-primary);
}

// 背景动画
.bg-animation {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.grid-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0% { transform: translateY(0); }
  100% { transform: translateY(50px); }
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.3;
  animation: float 20s ease-in-out infinite;
}

.orb-1 {
  width: 400px;
  height: 400px;
  background: var(--primary);
  top: -100px;
  left: -100px;
}

.orb-2 {
  width: 300px;
  height: 300px;
  background: var(--secondary);
  top: 50%;
  right: -100px;
  animation-delay: -7s;
}

.orb-3 {
  width: 350px;
  height: 350px;
  background: var(--accent);
  bottom: -100px;
  left: 30%;
  animation-delay: -14s;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(30px, 30px) scale(1.1); }
  50% { transform: translate(-20px, 50px) scale(0.9); }
  75% { transform: translate(40px, -20px) scale(1.05); }
}

// 全屏布局
.fullscreen-layout {
  position: relative;
  z-index: 1;
  width: 100%;
  height: 100%;
}

// 主布局
.main-layout {
  height: 100%;
  position: relative;
  z-index: 1;
}

// 侧边栏
.sidebar {
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  backdrop-filter: blur(20px);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  border-bottom: 1px solid var(--border);
}

.logo-icon {
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Orbitron', sans-serif;
  font-size: 24px;
  font-weight: 700;
  color: var(--bg-dark);
  box-shadow: 0 0 20px var(--primary-glow);
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.logo-text .title {
  font-family: 'Orbitron', sans-serif;
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 0.1em;
}

.logo-text .subtitle {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.3em;
}

.sidebar-menu {
  flex: 1;
  background: transparent;
  border: none;
  
  :deep(.el-menu-item) {
    margin: 4px 8px;
    border-radius: 8px;
    color: var(--text-secondary);
    
    &:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-primary);
    }
    
    &.is-active {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      color: var(--bg-dark);
      
      .el-icon {
        color: var(--bg-dark);
      }
    }
  }
}

.system-status {
  padding: 16px;
  border-top: 1px solid var(--border);
  
  .status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--text-muted);
    
    .el-icon {
      color: var(--primary);
    }
  }
  
  .memory-text {
    font-size: 11px;
    margin-left: auto;
  }
  
  :deep(.el-progress) {
    flex: 1;
    
    .el-progress-bar__outer {
      background: rgba(255, 255, 255, 0.1);
    }
    
    .el-progress-bar__inner {
      background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
  }
}

// 主内容
.main-content {
  padding: 24px;
  overflow-y: auto;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
  }
}

// 过渡动画
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>

<!-- 全局样式 - 不使用scoped -->
<style>
/* Tooltip 全局样式 */
.el-popper {
  background: #1e1e2e !important;
  color: #ffffff !important;
  border: 1px solid #3a3a4a !important;
}
.el-popper .el-popper__arrow::before {
  background: #1e1e2e !important;
  border-color: #3a3a4a !important;
}
.el-tooltip__popper {
  background: #1e1e2e !important;
  color: #ffffff !important;
  border: 1px solid #3a3a4a !important;
}
.el-tooltip__popper .el-popper__arrow::before {
  background: #1e1e2e !important;
  border-color: #3a3a4a !important;
}
</style>
