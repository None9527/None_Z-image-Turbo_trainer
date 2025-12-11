<template>
  <div class="training-config-page">
    <!-- 顶部配置管理栏 -->
    <div class="config-header glass-card">
      <div class="header-left">
        <h1><el-icon><Setting /></el-icon> 训练配置</h1>
        <div class="config-toolbar">
          <el-select v-model="currentConfigName" placeholder="选择配置..." @change="loadSavedConfig" style="width: 200px">
            <el-option label="默认配置" value="default" />
            <el-option v-for="cfg in savedConfigs.filter(c => c.name !== 'default')" :key="cfg.name" :label="cfg.name" :value="cfg.name" />
          </el-select>
          <el-button @click="showNewConfigDialog = true" :icon="Plus">新建</el-button>
          <el-button @click="showSaveAsDialog = true" :icon="Document">另存为</el-button>
          <el-button type="primary" @click="saveCurrentConfig" :loading="saving" :icon="Check">发送训练器</el-button>
          <el-button type="danger" @click="deleteCurrentConfig" :disabled="currentConfigName === 'default'" :icon="Delete">删除</el-button>
        </div>
      </div>
    </div>

    <!-- 新建配置对话框 -->
    <el-dialog v-model="showNewConfigDialog" title="新建配置" width="400px">
      <el-form label-width="80px">
        <el-form-item label="配置名称">
          <el-input v-model="newConfigName" placeholder="输入配置名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showNewConfigDialog = false">取消</el-button>
        <el-button type="primary" @click="createNewConfig">创建</el-button>
      </template>
    </el-dialog>

    <!-- 另存为对话框 -->
    <el-dialog v-model="showSaveAsDialog" title="另存为" width="400px">
      <el-form label-width="80px">
        <el-form-item label="配置名称">
          <el-input v-model="saveAsName" placeholder="输入新配置名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showSaveAsDialog = false">取消</el-button>
        <el-button type="primary" @click="saveAsNewConfig">保存</el-button>
      </template>
    </el-dialog>

    <!-- 配置内容 -->
    <el-card class="config-content-card glass-card" v-loading="loading">
      <el-collapse v-model="activeNames" class="config-collapse">

        <!-- 1. 模型类型选择 -->
        <el-collapse-item name="model">
          <template #title>
            <div class="collapse-title">
              <el-icon><Cpu /></el-icon>
              <span>模型类型</span>
              <el-tag :type="(modelTagType as 'primary' | 'success' | 'warning' | 'info' | 'danger')" size="small" style="margin-left: 10px">{{ modelDisplayName }}</el-tag>
            </div>
          </template>
          <div class="collapse-content">
            <div class="model-type-cards">
              <div 
                v-for="model in availableModels" 
                :key="model.value"
                :class="['model-card', { active: config.model_type === model.value, disabled: model.disabled }]"
                @click="!model.disabled && selectModelType(model.value)"
              >
                <div class="model-icon">{{ model.icon }}</div>
                <div class="model-info">
                  <div class="model-name">{{ model.label }}</div>
                  <div class="model-desc">{{ model.description }}</div>
                </div>
                <el-tag v-if="model.tag" :type="(model.tagType as 'primary' | 'success' | 'warning' | 'info' | 'danger')" size="small">{{ model.tag }}</el-tag>
              </div>
            </div>
          </div>
        </el-collapse-item>

        <!-- 2. 模型专属参数（根据模型类型显示） -->
        <el-collapse-item name="acrf">
          <template #title>
            <div class="collapse-title">
              <el-icon><DataAnalysis /></el-icon>
              <span>{{ config.model_type === 'zimage' ? 'Zimage 参数' : 'Longcat 参数' }}</span>
            </div>
          </template>
          <div class="collapse-content">
            <!-- Turbo 开关（两个模型都有） -->
            <div class="control-row">
              <span class="label">
                启用 Turbo
                <el-tooltip content="开启后使用加速推理模式，关闭则使用标准推理" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.acrf.enable_turbo" />
            </div>
            
            <!-- Turbo 步数（启用 Turbo 时显示） -->
            <div class="control-row" v-if="config.acrf.enable_turbo">
              <span class="label">
                Turbo 步数
                <el-tooltip content="生成时用多少步，这里就写多少步" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.acrf.turbo_steps" :min="1" :max="10" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.acrf.turbo_steps" :min="1" :max="10" :step="1" controls-position="right" class="input-fixed" />
            </div>

            <!-- ============ Zimage 特有参数 ============ -->
            <template v-if="config.model_type === 'zimage'">
              <div class="control-row">
                <span class="label">
                  Shift
                  <el-tooltip content="时间步偏移，影响噪声调度，默认 3.0" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.shift" :min="1" :max="5" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.shift" :min="1" :max="5" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">
                  Jitter Scale
                  <el-tooltip content="时间步抖动幅度，增加训练多样性，0=关闭" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.jitter_scale" :min="0" :max="0.1" :step="0.01" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.jitter_scale" :min="0" :max="0.1" :step="0.01" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- ============ Longcat 特有参数 ============ -->
            <template v-if="config.model_type === 'longcat'">
              <div class="control-row">
                <span class="label">
                  动态 Shift
                  <el-tooltip content="根据图像序列长度自动调整 shift 值，推荐开启" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-switch v-model="config.acrf.use_dynamic_shifting" />
              </div>
              <div class="control-row">
                <span class="label">
                  Base Shift
                  <el-tooltip content="动态 shift 的基础值，对应小图（默认 0.5）" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.base_shift" :min="0.1" :max="2" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.base_shift" :min="0.1" :max="2" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">
                  Max Shift
                  <el-tooltip content="动态 shift 的最大值，对应大图（默认 1.15）" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.max_shift" :min="0.5" :max="3" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.max_shift" :min="0.5" :max="3" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
            </template>
          </div>
        </el-collapse-item>

        <!-- 3. LoRA 配置 -->
        <el-collapse-item name="lora">
          <template #title>
            <div class="collapse-title">
              <el-icon><Grid /></el-icon>
              <span>LoRA 配置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="control-row">
              <span class="label">
                Network Dim (Rank)
                <el-tooltip content="LoRA 矩阵的秩，越大学习能力越强但文件越大，推荐 4-32" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.network.dim" :min="4" :max="128" :step="4" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.network.dim" :min="4" :max="128" :step="4" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                Network Alpha
                <el-tooltip content="缩放因子，通常设为 Dim 的一半，影响学习率效果" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.network.alpha" :min="1" :max="64" :step="0.5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.network.alpha" :min="1" :max="64" :step="0.5" controls-position="right" class="input-fixed" />
            </div>
            
            <!-- LoRA 高级选项 -->
            <div class="subsection-label">高级选项 (LoRA Targets)</div>
            <div class="control-row" v-if="config.model_type === 'zimage'">
              <span class="label">
                训练 AdaLN
                <el-tooltip content="训练 AdaLN 调制层 (激进模式，可能导致过拟合)" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.lora.train_adaln" />
            </div>
            <div class="control-row" v-if="config.model_type === 'longcat'">
              <span class="label">
                训练 Norm 层
                <el-tooltip content="训练 norm1.linear 和 norm1_context.linear" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.lora.train_norm" />
            </div>
            <div class="control-row" v-if="config.model_type === 'longcat'">
              <span class="label">
                训练单流层
                <el-tooltip content="训练单流 Transformer 块 (proj_mlp, proj_out)" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.lora.train_single_stream" />
            </div>
          </div>
        </el-collapse-item>

        <!-- 4. 训练设置 -->
        <el-collapse-item name="training">
          <template #title>
            <div class="collapse-title">
              <el-icon><TrendCharts /></el-icon>
              <span>训练设置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">输出设置 (OUTPUT)</div>
            <div class="form-row-full">
              <label>LoRA 输出名称</label>
              <el-input v-model="config.training.output_name" placeholder="zimage-lora" />
            </div>
            
            <div class="subsection-label">训练控制 (TRAINING CONTROL)</div>
            <div class="control-row">
              <span class="label">
                训练轮数
                <el-tooltip content="完整遍历数据集的次数，一般 5-20 轮即可" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.advanced.num_train_epochs" :min="1" :max="100" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.advanced.num_train_epochs" :min="1" :max="100" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                保存间隔
                <el-tooltip content="每隔几轮保存一次模型，便于挑选最佳效果" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.advanced.save_every_n_epochs" :min="1" :max="10" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.advanced.save_every_n_epochs" :min="1" :max="10" controls-position="right" class="input-fixed" />
            </div>

            <div class="subsection-label">优化器 (OPTIMIZER)</div>
            <div class="form-row-full">
              <label>
                优化器类型
                <el-tooltip content="AdamW8bit 省显存，Adafactor 更省但可能不稳定" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-select v-model="config.optimizer.type" style="width: 100%">
                <el-option label="AdamW" value="AdamW" />
                <el-option label="AdamW8bit (显存优化)" value="AdamW8bit" />
                <el-option label="Adafactor" value="Adafactor" />
              </el-select>
            </div>
            <div class="form-row-full">
              <label>
                学习率
                <el-tooltip content="模型学习的速度，太大会崩溃，太小学不到东西，推荐 1e-4" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-input v-model="config.training.learning_rate_str" placeholder="1e-4" @blur="parseLearningRate">
                <template #append>
                  <el-tooltip content="支持科学计数法，如 1e-4, 5e-5" placement="top">
                    <el-icon><InfoFilled /></el-icon>
                  </el-tooltip>
                </template>
              </el-input>
            </div>

            <div class="subsection-label">学习率调度器 (LR SCHEDULER)</div>
            <div class="form-row-full">
              <label>
                调度器类型
                <el-tooltip content="控制学习率变化方式，constant 最简单，cosine 后期更稳定" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-select v-model="config.training.lr_scheduler" style="width: 100%">
                <el-option label="constant (固定) ⭐推荐" value="constant" />
                <el-option label="linear (线性衰减)" value="linear" />
                <el-option label="cosine (余弦退火)" value="cosine" />
                <el-option label="cosine_with_restarts (余弦重启)" value="cosine_with_restarts" />
                <el-option label="constant_with_warmup (带预热)" value="constant_with_warmup" />
              </el-select>
            </div>
            <div class="control-row">
              <span class="label">
                Warmup Steps
                <el-tooltip content="预热步数。⚠️ 少样本训练建议设为 0，否则过长的预热会浪费训练时间（warmup 占比应 < 5%）" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.training.lr_warmup_steps" :min="0" :max="500" :step="5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.training.lr_warmup_steps" :min="0" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row" v-if="config.training.lr_scheduler === 'cosine_with_restarts'">
              <span class="label">
                Num Cycles
                <el-tooltip content="余弦重启周期数。cycles=1 时等同于普通 cosine；cycles=2+ 时学习率会在训练中重启（升高）" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.training.lr_num_cycles" :min="1" :max="5" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.training.lr_num_cycles" :min="1" :max="5" controls-position="right" class="input-fixed" />
            </div>

            <div class="subsection-label">梯度与内存 (GRADIENT & MEMORY)</div>
            <div class="control-row">
              <span class="label">
                梯度累积
                <el-tooltip content="模拟更大批次，显存不够时增大此值" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.advanced.gradient_accumulation_steps" :min="1" :max="16" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.advanced.gradient_accumulation_steps" :min="1" :max="16" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                梯度检查点
                <el-tooltip content="用计算换显存，开启可大幅节省显存" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.advanced.gradient_checkpointing" />
            </div>
            <div class="control-row">
              <span class="label">
                Blocks to Swap
                <el-tooltip content="将transformer blocks交换到CPU节省显存。16G显存建议4-8，24G可不设置" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-input-number v-model="config.advanced.blocks_to_swap" :min="0" :max="20" controls-position="right" style="width: 150px" />
            </div>
            <div class="form-row-full">
              <label>
                混合精度
                <el-tooltip content="bf16 推荐，fp16 兼容性更好，no 最精确但最慢" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-select v-model="config.advanced.mixed_precision" style="width: 100%">
                <el-option label="bf16 (推荐)" value="bf16" />
                <el-option label="fp16" value="fp16" />
                <el-option label="no (FP32)" value="no" />
              </el-select>
            </div>
            <div class="control-row">
              <span class="label">
                随机种子
                <el-tooltip content="固定种子可复现结果，不同种子效果略有差异" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
              </span>
              <el-input-number v-model="config.advanced.seed" :min="0" controls-position="right" style="width: 150px" />
            </div>
            
            <!-- 多卡训练配置 -->
            <div class="subsection-label">GPU 配置 (MULTI-GPU)</div>
            <div class="control-row">
              <span class="label">
                GPU 数量
                <el-tooltip content="使用多张 GPU 进行分布式训练，可加速训练但需要足够显存" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-select v-model="config.advanced.num_gpus" style="width: 150px">
                <el-option label="1 (单卡)" :value="1" />
                <el-option label="2" :value="2" />
                <el-option label="3" :value="3" />
                <el-option label="4" :value="4" />
                <el-option label="8" :value="8" />
              </el-select>
            </div>
            <div class="control-row">
              <span class="label">
                GPU ID
                <el-tooltip content="指定使用的 GPU 编号，如 '0,1' 或 '2,3'，留空自动选择" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-input v-model="config.advanced.gpu_ids" placeholder="如: 0,1,2" style="width: 150px" />
            </div>
          </div>
        </el-collapse-item>

        <!-- 5. 数据集配置 -->
        <el-collapse-item name="dataset">
          <template #title>
            <div class="collapse-title">
              <el-icon><Files /></el-icon>
              <span>数据集配置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">通用设置 (GENERAL)</div>
            <div class="control-row">
              <span class="label">
                批次大小
                <el-tooltip content="每次训练处理的图片数量，越大越快但显存占用越高" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.dataset.batch_size" :min="1" :max="16" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.dataset.batch_size" :min="1" :max="16" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                打乱数据
                <el-tooltip content="随机打乱训练顺序，避免模型记住顺序" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.dataset.shuffle" />
            </div>
            <div class="control-row">
              <span class="label">
                启用分桶
                <el-tooltip content="按图片尺寸分组，减少填充浪费，提高训练效率" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.dataset.enable_bucket" />
            </div>

            <div class="subsection-label-with-action">
              <span>数据集列表 (DATASETS)</span>
              <div class="dataset-toolbar">
                <el-select v-model="selectedDataset" placeholder="从数据集库选择..." clearable @change="onDatasetSelect" style="width: 280px">
                  <el-option v-for="ds in cachedDatasets" :key="ds.path" :label="ds.name" :value="ds.path">
                    <span style="float: left">{{ ds.name }}</span>
                    <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">{{ ds.files }} 文件</span>
                  </el-option>
                </el-select>
                <el-button size="small" type="primary" @click="addDataset" :icon="Plus">手动添加</el-button>
              </div>
            </div>
            
            <div v-if="config.dataset.datasets.length === 0" class="empty-datasets">
              <el-icon><FolderOpened /></el-icon>
              <p>暂无数据集，点击上方按钮添加</p>
            </div>

            <div v-for="(ds, idx) in config.dataset.datasets" :key="idx" class="dataset-item">
              <div class="dataset-header">
                <span class="dataset-index">数据集 {{ idx + 1 }}</span>
                <el-button type="danger" size="small" @click="removeDataset(idx)" :icon="Delete">删除</el-button>
              </div>
              <div class="form-row-full">
                <label>缓存目录路径</label>
                <el-input v-model="ds.cache_directory" placeholder="d:/AI/datasets/cache" />
              </div>
              <div class="control-row">
                <span class="label">
                  重复次数
                  <el-tooltip content="每张图片重复训练的次数，图片少时可增大" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="ds.num_repeats" :min="1" :max="100" :step="1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="ds.num_repeats" :min="1" :max="100" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">
                  分辨率上限
                  <el-tooltip content="图片最大分辨率，超过会缩小，越大显存占用越高" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="ds.resolution_limit" :min="256" :max="2048" :step="64" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="ds.resolution_limit" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
              </div>
            </div>
          </div>
        </el-collapse-item>

        <!-- 6. 高级选项 -->
        <el-collapse-item name="advanced">
          <template #title>
            <div class="collapse-title">
              <el-icon><Tools /></el-icon>
              <span>高级选项</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">SNR 参数（公用）</div>
            <div class="control-row">
              <span class="label">
                SNR Gamma
                <el-tooltip content="Min-SNR 截断值，平衡不同时间步的 loss 贡献，0=禁用，推荐 5.0" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.acrf.snr_gamma" :min="0" :max="10" :step="0.5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.acrf.snr_gamma" :min="0" :max="10" :step="0.5" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                SNR Floor
                <el-tooltip content="保底权重，确保高噪区（构图阶段）参与训练。10步模型关键参数，推荐 0.1" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.acrf.snr_floor" :min="0" :max="0.5" :step="0.01" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.acrf.snr_floor" :min="0" :max="0.5" :step="0.01" controls-position="right" class="input-fixed" />
            </div>

            <div class="subsection-label">MSE/L2 混合损失（构图增强）</div>
            <div class="control-row">
              <span class="label">
                启用 L2 混合
                <el-tooltip content="同batch混合锚点流+自由流L2损失，增强构图学习能力" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.acrf.raft_mode" />
            </div>
            <template v-if="config.acrf.raft_mode">
              <div class="control-row">
                <span class="label">
                  调度模式
                  <el-tooltip content="L2 比例随训练进度变化的方式" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-select v-model="config.acrf.l2_schedule_mode" style="width: 160px">
                  <el-option value="constant" label="固定值" />
                  <el-option value="linear_increase" label="渐进增加 (适合蒸馏)" />
                  <el-option value="linear_decrease" label="渐进减少 (适合Turbo)" />
                  <el-option value="step" label="自定义阶梯" />
                </el-select>
              </div>
              <div class="control-row">
                <span class="label">
                  起始比例
                  <el-tooltip content="训练开始时的 L2 比例" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.l2_initial_ratio" :min="0.05" :max="1.0" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.l2_initial_ratio" :min="0.05" :max="1.0" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" v-if="config.acrf.l2_schedule_mode !== 'constant'">
                <span class="label">
                  结束比例
                  <el-tooltip content="训练结束时的 L2 比例" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.l2_final_ratio" :min="0.05" :max="1.0" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.l2_final_ratio" :min="0.05" :max="1.0" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" v-if="config.acrf.l2_schedule_mode === 'step'">
                <span class="label">
                  阶梯切换 Epoch
                  <el-tooltip content="在哪些 epoch 切换 L2 比例 (逗号分隔，如 3,6)" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-input v-model="config.acrf.l2_milestones" placeholder="3,6" style="width: 120px" />
              </div>
              <div class="control-row">
                <span class="label">
                  L2 包含锚点
                  <el-tooltip content="L2 损失同时计算锚点时间步，使 L2 覆盖全部时间步" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-switch v-model="config.acrf.l2_include_anchor" />
              </div>
              <div class="control-row" v-if="config.acrf.l2_include_anchor">
                <span class="label">
                  L2 锚点比例
                  <el-tooltip content="锚点时间步的 L2 损失权重。这是与 L1 叠加的权重" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.acrf.l2_anchor_ratio" :min="0.05" :max="1.0" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.acrf.l2_anchor_ratio" :min="0.05" :max="1.0" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <div class="subsection-label">Latent Jitter（构图突破）</div>
            <div class="control-row">
              <span class="label">
                Latent Jitter Scale
                <el-tooltip content="在 x_t 上添加空间抠动，垂直于流线，真正改变构图的关键。推荐 0.03-0.05" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.acrf.latent_jitter_scale" :min="0" :max="0.1" :step="0.01" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.acrf.latent_jitter_scale" :min="0" :max="0.1" :step="0.01" controls-position="right" class="input-fixed" />
            </div>

            <div class="subsection-label">损失权重配置（自由组合）</div>
            
            <!-- 基础损失: L1 + Cosine (始终显示) -->
            <div class="control-row">
              <span class="label">
                Lambda L1 (Charbonnier)
                <el-tooltip content="基础L1损失权重，平滑的像素距离损失" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.training.lambda_l1" :min="0" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.training.lambda_l1" :min="0" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                Lambda Cosine
                <el-tooltip content="方向一致性损失权重，约束速度方向正确" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.training.lambda_cosine" :min="0" :max="1" :step="0.05" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.training.lambda_cosine" :min="0" :max="1" :step="0.05" controls-position="right" class="input-fixed" />
            </div>
            
            <!-- 频域感知损失 (开关+权重+子参数) -->
            <div class="subsection-label">频域感知损失</div>
            <div class="control-row">
              <span class="label">
                启用频域感知
                <el-tooltip content="锐化细节纹理，增强高频信息" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.training.enable_freq" />
            </div>
            <template v-if="config.training.enable_freq">
              <div class="control-row">
                <span class="label">
                  混合权重 (λ_freq)
                  <el-tooltip content="与基础损失混合的比例" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.training.lambda_freq" :min="0.1" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_freq" :min="0.1" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 高频权重 (alpha_hf)</span>
                <el-slider v-model="config.training.alpha_hf" :min="0" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.alpha_hf" :min="0" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 低频权重 (beta_lf)</span>
                <el-slider v-model="config.training.beta_lf" :min="0" :max="1" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.beta_lf" :min="0" :max="1" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
            </template>
            
            <!-- 风格结构损失 (开关+权重+子参数) -->
            <div class="subsection-label">风格结构损失</div>
            <div class="control-row">
              <span class="label">
                启用风格结构
                <el-tooltip content="学习光影色调、结构布局" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="config.training.enable_style" />
            </div>
            <template v-if="config.training.enable_style">
              <div class="control-row">
                <span class="label">
                  混合权重 (λ_style)
                  <el-tooltip content="与基础损失混合的比例" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="config.training.lambda_style" :min="0.1" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_style" :min="0.1" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 结构锁 (λ_struct)</span>
                <el-slider v-model="config.training.lambda_struct" :min="0" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_struct" :min="0" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 光影 (λ_light)</span>
                <el-slider v-model="config.training.lambda_light" :min="0" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_light" :min="0" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 色调 (λ_color)</span>
                <el-slider v-model="config.training.lambda_color" :min="0" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_color" :min="0" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 质感 (λ_tex)</span>
                <el-slider v-model="config.training.lambda_tex" :min="0" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="config.training.lambda_tex" :min="0" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
            </template>



            <div class="subsection-label">其他高级参数</div>
            <div class="control-row">
              <span class="label">
                Max Grad Norm
                <el-tooltip content="梯度裁剪阈值，防止梯度爆炸，一般保持默认" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.advanced.max_grad_norm" :min="0" :max="20" :step="0.5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.advanced.max_grad_norm" :min="0" :max="20" :step="0.5" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                Weight Decay
                <el-tooltip content="权重衰减，防止过拟合，一般保持0即可" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="config.training.weight_decay" :min="0" :max="0.1" :step="0.001" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="config.training.weight_decay" :min="0" :step="0.001" controls-position="right" class="input-fixed" :precision="3" />
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { Setting, Refresh, Check, FolderOpened, DataAnalysis, Grid, TrendCharts, Files, Tools, Plus, Delete, Document, InfoFilled, QuestionFilled, Promotion, Cpu } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

const route = useRoute()

const activeNames = ref(['model', 'acrf', 'lora', 'training', 'dataset', 'advanced'])
const loading = ref(false)
const saving = ref(false)
const selectedPreset = ref('')
const presets = ref<any[]>([])

// Config management
const currentConfigName = ref('default')
const savedConfigs = ref<any[]>([])
const showNewConfigDialog = ref(false)
const showSaveAsDialog = ref(false)
const newConfigName = ref('')
const saveAsName = ref('')

// Dataset management
const cachedDatasets = ref<any[]>([])
const selectedDataset = ref('')

// System paths (read-only, from env)
const systemPaths = ref({
  model_path: '',
  output_base_dir: ''
})

// 可用模型列表
type TagType = 'primary' | 'success' | 'warning' | 'info' | 'danger'

const availableModels = ref<Array<{
  value: string
  label: string
  icon: string
  description: string
  tag: string
  tagType: TagType
  disabled: boolean
}>>([
  {
    value: 'zimage',
    label: 'Z-Image (Turbo)',
    icon: '⚡',
    description: '10 步加速推理，原生 Turbo 模型',
    tag: '推荐',
    tagType: 'success',
    disabled: false
  },
  {
    value: 'longcat',
    label: 'LongCat-Image',
    icon: '🐱',
    description: '基于 FLUX 架构，高质量生成',
    tag: '新',
    tagType: 'warning',
    disabled: false
  }
])

// 模型类型显示
const modelDisplayName = computed(() => {
  const model = availableModels.value.find(m => m.value === config.value.model_type)
  return model?.label || 'Z-Image'
})

const modelTagType = computed((): TagType => {
  const model = availableModels.value.find(m => m.value === config.value.model_type)
  return model?.tagType || 'primary'
})

function selectModelType(type: string) {
  config.value.model_type = type
}

// 默认配置结构
function getDefaultConfig() {
  return {
    name: 'default',
    model_type: 'zimage',  // 模型类型
    acrf: {
      enable_turbo: true,  // Turbo 开关
      turbo_steps: 10,
      // Zimage 参数
      shift: 3.0,
      jitter_scale: 0.02,
      // Longcat 动态 shift 参数
      use_dynamic_shifting: true,
      base_shift: 0.5,
      max_shift: 1.15,
      // Min-SNR 加权参数（公用）
      snr_gamma: 5.0,
      snr_floor: 0.1,
      use_anchor: true,
      // MSE/L2 混合模式参数
      raft_mode: false,
      free_stream_ratio: 0.3,
      // L2 调度参数
      l2_schedule_mode: 'constant',
      l2_initial_ratio: 0.3,
      l2_final_ratio: 0.3,
      l2_milestones: '',
      l2_include_anchor: false,
      l2_anchor_ratio: 0.3,
      // Latent Jitter (构图突破)
      latent_jitter_scale: 0.0
    },
    network: {
      dim: 8,
      alpha: 4.0
    },
    lora: {
      train_adaln: false,
      train_norm: false,
      train_single_stream: false
    },
    optimizer: {
      type: 'AdamW8bit',
      learning_rate: '1e-4'
    },
    training: {
      output_name: 'zimage-lora',
      learning_rate: 0.0001,
      learning_rate_str: '1e-4',  // 用于UI显示
      weight_decay: 0,
      lr_scheduler: 'constant',
      lr_warmup_steps: 0,
      lr_num_cycles: 1,
      // 基础损失权重
      lambda_l1: 1.0,
      lambda_cosine: 0.1,
      // 频域感知 (开关+权重+子参数)
      enable_freq: false,
      lambda_freq: 0.3,
      alpha_hf: 1.0,
      beta_lf: 0.2,
      // 风格结构 (开关+权重+子参数)
      enable_style: false,
      lambda_style: 0.3,
      lambda_struct: 1.0,
      lambda_light: 0.5,
      lambda_color: 0.3,
      lambda_tex: 0.5,
      // 兼容旧参数
      lambda_fft: 0
    },
    dataset: {
      batch_size: 1,
      shuffle: true,
      enable_bucket: true,
      datasets: [] as any[]
    },
    advanced: {
      max_grad_norm: 1.0,
      gradient_checkpointing: true,
      blocks_to_swap: 0,
      num_train_epochs: 10,
      save_every_n_epochs: 1,
      gradient_accumulation_steps: 4,
      mixed_precision: 'bf16',
      seed: 42,
      num_gpus: 1,
      gpu_ids: ''
    }
  }
}

const config = ref(getDefaultConfig())

onMounted(async () => {
  await loadConfigList()
  
  // 检查 URL 参数，如果有 edit 参数则加载对应配置
  const editConfig = route.query.edit as string
  if (editConfig && editConfig !== 'default') {
    await loadConfig(editConfig)
    ElMessage.info(`正在编辑配置: ${editConfig}`)
  } else {
  await loadConfig('default')
  }
  
  await loadPresets()
  await loadCachedDatasets()
})

// Load list of saved configs
async function loadConfigList() {
  try {
    const res = await axios.get('/api/training/configs')
    savedConfigs.value = res.data.configs
  } catch (e) {
    console.error('Failed to load config list:', e)
  }
}

// Load a specific config
async function loadConfig(configName: string) {
  loading.value = true
  try {
    const res = await axios.get(`/api/training/config/${configName}`)
    const defaultCfg = getDefaultConfig()
    // 深度合并，确保所有字段都有值
    config.value = {
      ...defaultCfg,
      ...res.data,
      acrf: { ...defaultCfg.acrf, ...res.data.acrf },
      network: { ...defaultCfg.network, ...res.data.network },
      optimizer: { ...defaultCfg.optimizer, ...res.data.optimizer },
      training: { ...defaultCfg.training, ...res.data.training },
      dataset: { 
        ...defaultCfg.dataset, 
        ...res.data.dataset,
        datasets: res.data.dataset?.datasets || []
      },
      advanced: { ...defaultCfg.advanced, ...res.data.advanced }
    }
    // 初始化学习率字符串
    const lr = config.value.training.learning_rate || 0.0001
    config.value.training.learning_rate_str = lr >= 0.001 ? lr.toString() : lr.toExponential()
    currentConfigName.value = configName
  } catch (e: any) {
    ElMessage.error('加载配置失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loading.value = false
  }
}

// Load from saved config (from dropdown)
async function loadSavedConfig() {
  if (currentConfigName.value) {
    await loadConfig(currentConfigName.value)
  }
}

// Save current config
async function saveCurrentConfig() {
  if (!currentConfigName.value) {
    ElMessage.warning('请先选择或创建一个配置')
    return
  }
  
  saving.value = true
  try {
    await axios.post('/api/training/config/save', {
      name: currentConfigName.value,
      config: config.value
    })
    ElMessage.success('配置已发送到训练器')
    await loadConfigList()
  } catch (e: any) {
    ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    saving.value = false
  }
}

// Create new config
async function createNewConfig() {
  if (!newConfigName.value.trim()) {
    ElMessage.warning('请输入配置名称')
    return
  }
  
  try {
    await axios.post('/api/training/config/save', {
      name: newConfigName.value,
      config: { ...config.value, name: newConfigName.value }
    })
    ElMessage.success(`配置 "${newConfigName.value}" 已创建`)
    currentConfigName.value = newConfigName.value
    await loadConfigList()
    showNewConfigDialog.value = false
    newConfigName.value = ''
  } catch (e: any) {
    ElMessage.error('创建失败: ' + (e.response?.data?.detail || e.message))
  }
}

// Save as new config
async function saveAsNewConfig() {
  if (!saveAsName.value.trim()) {
    ElMessage.warning('请输入配置名称')
    return
  }
  
  try {
    await axios.post('/api/training/config/save', {
      name: saveAsName.value,
      config: { ...config.value, name: saveAsName.value }
    })
    ElMessage.success(`已另存为 "${saveAsName.value}"`)
    currentConfigName.value = saveAsName.value
    await loadConfigList()
    showSaveAsDialog.value = false
    saveAsName.value = ''
  } catch (e: any) {
    ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message))
  }
}

// Delete current config
async function deleteCurrentConfig() {
  if (currentConfigName.value === 'default') {
    return
  }
  
  try {
    await ElMessageBox.confirm(
      `确定要删除配置 "${currentConfigName.value}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await axios.delete(`/api/training/config/${currentConfigName.value}`)
    ElMessage.success('配置已删除')
    currentConfigName.value = 'default'
    await loadConfigList()
    await loadConfig('default')
  } catch (e: any) {
    if (e !== 'cancel') {
      ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message))
    }
  }
}

// Load presets
async function loadPresets() {
  try {
    const res = await axios.get('/api/training/presets')
    presets.value = res.data.presets
  } catch (e) {
    console.error('Failed to load presets:', e)
  }
}

// Load preset
function loadPreset() {
  if (!selectedPreset.value) return
  
  const preset = presets.value.find(p => p.name === selectedPreset.value)
  if (preset) {
    config.value = JSON.parse(JSON.stringify(preset.config))
    ElMessage.success(`已加载预设: ${preset.name}`)
    selectedPreset.value = ''
  }
}

// Load cached datasets
async function loadCachedDatasets() {
  try {
    const res = await axios.get('/api/dataset/cached')
    cachedDatasets.value = res.data.datasets
  } catch (e) {
    console.error('Failed to load cached datasets:', e)
  }
}

// Add dataset (from selector)
function onDatasetSelect() {
  if (selectedDataset.value) {
    addDatasetFromCache(selectedDataset.value)
    selectedDataset.value = ''
  }
}

function addDatasetFromCache(datasetPath: string) {
  config.value.dataset.datasets.push({
    cache_directory: datasetPath,
    num_repeats: 1,
    resolution_limit: 1024
  })
}

// Manual add dataset
function addDataset() {
  config.value.dataset.datasets.push({
    cache_directory: '',
    num_repeats: 1,
    resolution_limit: 1024
  })
}

// Remove dataset
function removeDataset(idx: number) {
  config.value.dataset.datasets.splice(idx, 1)
}

// 解析学习率（支持科学计数法）
function parseLearningRate() {
  const str = config.value.training.learning_rate_str
  if (!str) return
  
  try {
    const value = parseFloat(str)
    if (!isNaN(value) && value > 0) {
      config.value.training.learning_rate = value
    }
  } catch (e) {
    console.warn('Invalid learning rate:', str)
  }
}

// 格式化学习率为字符串
function formatLearningRate(value: number): string {
  if (value >= 0.001) return value.toString()
  return value.toExponential().replace('e-', 'e-').replace('+', '')
}
</script>

<style scoped>
.training-config-page {
  padding: 24px;
  height: 100%;
  overflow-y: auto;
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 20px 24px;
  margin: 0 auto 24px auto;
  max-width: 1000px;
}

.header-left {
  flex: 1;
}

.header-left h1 {
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 24px;
}

.config-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
}

.dataset-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
}

.config-path {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.config-content-card {
  max-width: 1000px;
  margin: 0 auto;
}

.config-collapse {
  border: none !important;
}

.config-collapse :deep(.el-collapse-item) {
  margin-bottom: 16px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  overflow: hidden;
  background-color: var(--el-bg-color);
}

.config-collapse :deep(.el-collapse-item:last-child) {
  margin-bottom: 0;
}

.config-collapse :deep(.el-collapse-item__header) {
  background-color: var(--el-fill-color-lighter);
  padding: 16px 20px;
  font-weight: bold;
  border-bottom: 1px solid transparent;
  height: auto;
  line-height: 1.5;
}

.config-collapse :deep(.el-collapse-item.is-active .el-collapse-item__header) {
  border-bottom-color: var(--el-border-color-lighter);
}

.config-collapse :deep(.el-collapse-item__wrap) {
  border: none;
}

/* Force hide content when item is not active */
.config-collapse :deep(.el-collapse-item:not(.is-active) .el-collapse-item__wrap) {
  display: none !important;
}

.config-collapse :deep(.el-collapse-item__content) {
  padding: 0 0 16px 0;
}

.collapse-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 15px;
}

.collapse-content {
  padding: 16px 20px 0 20px;
  max-width: 800px;
}

.subsection-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--el-text-color-secondary);
  margin: 20px 0 12px 0;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding-top: 20px;
  border-top: 1px solid var(--el-border-color-lighter);
}

.subsection-label:first-child {
  margin-top: 16px;
  padding-top: 0;
  border-top: none;
}

.subsection-label-with-action {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  font-weight: 700;
  color: var(--el-text-color-secondary);
  margin: 20px 0 12px 0;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding-top: 20px;
  border-top: 1px solid var(--el-border-color-lighter);
}

.form-row-full {
  margin-bottom: 16px;
}

.form-row-full:last-child {
  margin-bottom: 0;
}

.form-row-full label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--el-text-color-regular);
  margin-bottom: 6px;
}

.readonly-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: var(--el-color-info-light-9);
  border-left: 3px solid var(--el-color-info);
  border-radius: 4px;
  font-size: 12px;
  color: var(--el-color-info);
  margin-bottom: 16px;
}

.readonly-row {
  background: var(--el-fill-color-lighter);
  padding: 12px;
  border-radius: 6px;
}

.readonly-value {
  font-family: monospace;
  font-size: 13px;
  color: var(--el-text-color-primary);
  padding: 4px 0;
}

.control-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.control-row:last-child {
  margin-bottom: 0;
}

.control-row .label {
  font-size: 12px;
  color: var(--el-text-color-regular);
  width: 160px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 4px;
}

.help-icon {
  color: var(--el-color-primary-light-3);
  cursor: help;
  font-size: 14px;
  opacity: 0.8;
}

.help-icon:hover {
  color: var(--el-color-primary);
  opacity: 1;
}

.form-row-full label .help-icon {
  margin-left: 4px;
}


.slider-flex {
  flex: 1;
  margin-right: 8px;
}

.input-fixed {
  width: 100px !important;
}

.dataset-item {
  background: var(--el-bg-color);
  padding: 16px;
  border-radius: 6px;
  border: 1px solid var(--el-border-color-light);
  margin-bottom: 12px;
}

.dataset-item:last-child {
  margin-bottom: 0;
}

.dataset-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.dataset-index {
  font-weight: bold;
  font-size: 13px;
  color: var(--el-color-primary);
}

.empty-datasets {
  text-align: center;
  padding: 40px 20px;
  color: var(--el-text-color-secondary);
}

.empty-datasets .el-icon {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.5;
}

.empty-datasets p {
  margin: 0;
  font-size: 13px;
}

/* 模型类型卡片样式 */
.model-type-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
}

.model-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border: 2px solid var(--el-border-color-lighter);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: var(--el-bg-color);
}

.model-card:hover:not(.disabled) {
  border-color: var(--el-color-primary-light-5);
  background: var(--el-color-primary-light-9);
}

.model-card.active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
  box-shadow: 0 0 0 3px var(--el-color-primary-light-7);
}

.model-card.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.model-icon {
  font-size: 32px;
  flex-shrink: 0;
}

.model-info {
  flex: 1;
  min-width: 0;
}

.model-name {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
}

.model-desc {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.4;
}
</style>
