<!-- 新增：模型检测抽象层面板 -->
    <div class="status-card">
      <div class="card-header">
        <el-icon><Monitor /></el-icon>
        <span>模型检测与管理（新）</span>
        <el-tag type="info" size="small" effect="plain">
          抽象层
        </el-tag>
      </div>
      <div class="abstract-layer-panel">
        <p class="panel-description">
          使用新的模型检测和下载抽象层，支持多模型自动检测、完整性验证和统一管理。
        </p>
        <div class="panel-actions">
          <el-button type="primary" @click="goToModelDetection">
            <el-icon><Monitor /></el-icon>
            进入模型检测面板
          </el-button>
          <el-button @click="refreshAbstractLayer">
            <el-icon><Refresh /></el-icon>
            刷新状态
          </el-button>
        </div>
        
        <!-- 快速状态概览 -->
        <div class="quick-status" v-if="abstractLayerStatus">
          <el-descriptions :column="2" size="small" border>
            <el-descriptions-item label="Z-Image状态">
              <el-tag :type="getAbstractStatusType('zimage')" size="small">
                {{ getAbstractStatusText('zimage') }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="LongCat状态">
              <el-tag :type="getAbstractStatusType('longcat')" size="small">
                {{ getAbstractStatusText('longcat') }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </div>