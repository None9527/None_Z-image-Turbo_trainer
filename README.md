# None Trainer

<div align="center">

![Logo](https://img.shields.io/badge/None-Trainer-f0b429?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMxYTFhMWQiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJtMTIgMyA4IDR2NmMwIDUuNTMtMy42MSA4Ljk5LTggMTEtNC4zOS0yLjAxLTgtNS40Ny04LTExVjdsMTItNFoiLz48L3N2Zz4=)

**Z-Image Turbo LoRA 训练工作室**

基于 **AC-RF（锚点耦合整流流）** 算法的高效 LoRA 微调工具

</div>

---

## ✨ 特性

| 特性 | 说明 |
|------|------|
| 🎯 **锚点耦合采样** | 只在关键时间步训练，高效稳定 |
| ⚡ **10步快速推理** | 保持 Turbo 模型的加速结构 |
| 📉 **Min-SNR 加权** | 减少不同时间步的 loss 波动 |
| 🎨 **多种损失模式** | 频域感知 / 风格结构 / 统一模式 |
| 🔧 **自动硬件优化** | 检测 GPU 并自动配置 (Tier S/A/B) |
| 🖥️ **现代化 WebUI** | Vue.js + FastAPI 全栈界面 |
| 📊 **实时监控** | Loss 曲线、进度、显存监控 |
| 🏷️ **Ollama 标注** | 一键 AI 图片打标 |

---

## 🚀 快速开始

### Step 1: 安装 PyTorch（必须）

根据你的 CUDA 版本选择：

```bash
# CUDA 12.8 (RTX 40系列推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (旧显卡)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: 安装 Flash Attention（推荐）

Flash Attention 可显著降低显存占用并加速训练。

**Linux** - 从 [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) 下载：

```bash
# 查看你的环境版本
python --version                                      # 例如: Python 3.12
python -c "import torch; print(torch.version.cuda)"  # 例如: 12.8

# 下载对应版本（示例：Python 3.12 + CUDA 12 + PyTorch 2.5）
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 安装
pip install flash_attn-*.whl
```

**Windows** - 从 [AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl/tree/main) 下载预编译版：

```batch
:: 示例：Python 3.12 + CUDA 12.8 + PyTorch 2.9.1
pip install https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl

:: 或下载后本地安装
pip install flash_attn-xxx.whl
```

> **提示**: 如果没有对应版本，可跳过此步，程序会自动使用 SDPA 作为备选。

### Step 3: 安装 Diffusers（必须）

⚠️ **注意**: 本项目需要 diffusers 0.36+（开发版），pip 暂无发布，需从 git 安装：

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### Step 4: 一键部署

#### Linux / Mac

```bash
# 克隆项目
git clone https://github.com/None9527/None_Z-image-Turbo_trainer.git
cd None_Z-image-Turbo_trainer

# 一键安装依赖
chmod +x setup.sh
./setup.sh

# 编辑配置（设置模型路径）
cp env.example .env
nano .env

# 启动服务
./start.sh
```

#### Windows

```batch
:: 克隆项目
git clone https://github.com/None9527/None_Z-image-Turbo_trainer.git
cd None_Z-image-Turbo_trainer

:: 一键安装依赖（双击或命令行）
setup.bat

:: 编辑配置（设置模型路径）
copy env.example .env
notepad .env

:: 启动服务
start.bat
```

### Step 5: 访问 Web UI

部署完成后打开浏览器访问: **http://localhost:9198**

---

## 📦 手动安装（可选）

<details>
<summary>如果一键部署遇到问题，可展开手动安装</summary>

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装 diffusers 最新版
pip install git+https://github.com/huggingface/diffusers.git

# 3. 安装本项目
pip install -e .

# 4. 创建配置文件
cp env.example .env

# 5. 启动服务
cd webui-vue/api && python main.py --port 9198
```

</details>

---

## 🖥️ 命令行使用（高级）

除了 Web UI，你也可以直接使用命令行进行操作：

### 生成缓存

```bash
# 生成 Latent 缓存（VAE 编码）
python -m zimage_trainer.cache_latents \
    --model_path ./zimage_models \
    --dataset_path ./datasets/your_dataset \
    --output_dir ./datasets/your_dataset

# 生成 Text 缓存（文本编码）
python -m zimage_trainer.cache_text_encoder \
    --text_encoder ./zimage_models/text_encoder \
    --input_dir ./datasets/your_dataset \
    --output_dir ./datasets/your_dataset \
    --max_length 512  # 可选: 256/512/1024, 默认 512
```

### 启动训练

```bash
# 使用配置文件训练（推荐）
python scripts/train_acrf.py --config config/acrf_config.toml

# 指定损失模式
python scripts/train_acrf.py --config config/acrf_config.toml --loss_mode frequency

# 频域感知模式 + 自定义参数
python scripts/train_acrf.py --config config/acrf_config.toml \
    --loss_mode frequency \
    --alpha_hf 1.0 \
    --beta_lf 0.2

# 风格结构模式
python scripts/train_acrf.py --config config/acrf_config.toml \
    --loss_mode style \
    --lambda_struct 1.0 \
    --lambda_light 0.5 \
    --lambda_color 0.3

# 统一模式（频域 + 风格）
python scripts/train_acrf.py --config config/acrf_config.toml --loss_mode unified
```

### 推理生成

```bash
# 加载 LoRA 生成图片
python -m zimage_trainer.inference \
    --model_path ./zimage_models \
    --lora_path ./output/your_lora.safetensors \
    --prompt "your prompt here" \
    --output_path ./output/generated.png \
    --num_inference_steps 10
```

### 启动 Web UI 服务

```bash
# 方式一：使用脚本
./start.sh          # Linux/Mac
start.bat           # Windows

# 方式二：直接启动
cd webui-vue/api
python main.py --port 9198 --host 0.0.0.0

# 方式三：使用 uvicorn（支持热重载）
cd webui-vue/api
uvicorn main:app --port 9198 --reload
```

### 转换 LoRA 格式

```bash
# 转换为 ComfyUI 兼容格式
python scripts/convert_lora_comfyui.py \
    --input ./output/your_lora.safetensors \
    --output ./output/your_lora_comfyui.safetensors
```

---

## ⚙️ 配置说明

### 环境变量 (`.env`)

```bash
# 服务配置
TRAINER_PORT=9198           # Web UI 端口
TRAINER_HOST=0.0.0.0        # 监听地址

# 模型路径
MODEL_PATH=/./zimage_models

# 数据集路径
DATASET_PATH=./datasets

# Ollama 配置
OLLAMA_HOST=http://127.0.0.1:11434
```

### 训练参数 (`config/acrf_config.toml`)

```toml
[acrf]
turbo_steps = 10        # 锚点数（推理步数）
shift = 3.0             # Z-Image 官方值
jitter_scale = 0.02     # 锚点抖动

[lora]
network_dim = 16        # LoRA rank
network_alpha = 16      # LoRA alpha

[training]
learning_rate = 1e-4    # 学习率
num_train_epochs = 10   # 训练轮数
snr_gamma = 5.0         # Min-SNR 加权
loss_mode = "standard"  # 损失模式（见下方说明）

[dataset]
batch_size = 1
enable_bucket = true
max_sequence_length = 512  # 文本序列长度 (需与缓存一致)
```

### 🎨 损失模式 (Loss Mode)

新版本支持 4 种损失模式，可在前端"高级选项"中选择：

| 模式 | 说明 | 适用场景 | 推荐参数 |
|------|------|----------|----------|
| **standard** | 基础 MSE + 可选 FFT/Cosine | 通用训练 | 默认即可 |
| **frequency** | 频域感知（高频L1 + 低频Cosine） | 锐化细节，不改风格 | alpha_hf=1.0, beta_lf=0.2 |
| **style** | 风格结构（SSIM + Lab统计量） | 学习大师光影/调色 | lambda_struct=1.0 |
| **unified** | 频域 + 风格 组合 | 全面增强 | 两者默认值 |

#### 频域感知模式 (frequency)

```
核心原理：
┌─────────────────────────────────────────┐
│  Latent ──► 降采样 ──► 低频（结构）     │
│         └──► 高频 = 原始 - 低频（细节） │
│                                         │
│  Loss = MSE + α·L1(高频) + β·Cos(低频)  │
└─────────────────────────────────────────┘

参数说明：
- alpha_hf: 高频增强权重（↑锐化 ↑噪点风险）推荐 0.5~1.0
- beta_lf:  低频锁定权重（↑保持结构）推荐 0.1~0.3
```

#### 风格结构模式 (style)

```
核心原理：
┌─────────────────────────────────────────────┐
│  Latent 近似 Lab 空间                        │
│  ├─ L通道 ──► SSIM（锁结构）                │
│  │         ├─ Mean/Std（学光影）            │
│  │         └─ 高频L1（学纹理）              │
│  └─ ab通道 ──► Mean/Std（学色调）           │
└─────────────────────────────────────────────┘

参数说明：
- lambda_struct: 结构锁（防脸崩）推荐 0.5~1.0
- lambda_light:  光影学习（学S曲线）推荐 0.3~0.8
- lambda_color:  色调迁移（学冷暖调）推荐 0.2~0.5
- lambda_tex:    质感增强（学颗粒感）推荐 0.3~0.5
```

#### 选择建议

| 你的目标 | 推荐模式 |
|----------|----------|
| 训练人物/角色 LoRA | `standard` |
| 提升画面清晰度 | `frequency` |
| 学习特定摄影师风格 | `style` |
| 全面提升质量 | `unified` |

> 💡 **新手建议**：先用 `standard` 模式训练，效果不满意再尝试其他模式。

### 硬件分级

| Tier | 显存 | 显卡示例 | 自动优化策略 |
|------|------|----------|-------------|
| **S** | 32GB+ | A100/H100/5090 | 全性能，无压缩 |
| **A** | 24GB | 3090/4090 | 高性能，原生 SDPA |
| **B** | 16GB | 4080/4070Ti | 平衡模式，轻度压缩 |

---

## 📊 使用流程

| 步骤 | 功能 | 说明 |
|:---:|:---:|:---|
| 1️⃣ | **数据集** | 导入图片、Ollama AI 标注 |
| ➡️ | | |
| 2️⃣ | **缓存** | 预计算 Latent 和 Text 嵌入 |
| ➡️ | | |
| 3️⃣ | **训练** | AC-RF LoRA 微调 |
| ➡️ | | |
| 4️⃣ | **生成** | 加载 LoRA 测试效果 |

---

## 🔧 常见问题

<details>
<summary><strong>Q: loss 跳动很大（0.08-0.6）？</strong></summary>

A: 正常现象！不同 sigma 下预测难度不同。看 **EMA loss** 是否整体下降即可。

</details>

<details>
<summary><strong>Q: CUDA Out of Memory？</strong></summary>

A: 尝试以下方法：
- 增大 `gradient_accumulation_steps`（如 4 → 8）
- 降低 `network_dim`（如 32 → 16）
- 确保已安装 Flash Attention

</details>

<details>
<summary><strong>Q: 训练多少 epoch？</strong></summary>

A: 取决于数据集大小：
- < 50 张：10-15 epoch
- 50-200 张：8-10 epoch
- \> 200 张：5-8 epoch

</details>

---

## 📬 联系方式

- 📧 lihaonan1082@gmail.com
- 📮 592532681@qq.com

---

## 📝 License

Apache 2.0

## 🙏 致谢

- [Z-Image](https://github.com/Alpha-VLLM/Lumina-Image) - 基础模型
- [diffusers](https://github.com/huggingface/diffusers) - 训练框架
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - 高效注意力
  
---

<div align="center">

**Made with ❤️ by None**

</div>
