# None Trainer

<div align="center">

![Logo](https://img.shields.io/badge/None-Trainer-f0b429?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMxYTFhMWQiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJtMTIgMyA4IDR2NmMwIDUuNTMtMy42MSA4Ljk5LTggMTEtNC4zOS0yLjAxLTgtNS40Ny04LTExVjdsMTItNFoiLz48L3N2Zz4=)

**Z-Image / LongCat-Image LoRA Training Studio**

Efficient LoRA fine-tuning tool based on **AC-RF (Anchor-Coupled Rectified Flow)** algorithm

Supports: **Z-Image Turbo** | **LongCat-Image**

[中文版 README](README.md)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Anchor-Coupled Sampling** | Train only at key timesteps, efficient and stable |
| ⚡ **10-Step Fast Inference** | Maintains Turbo model's acceleration structure |
| 📉 **Min-SNR Weighting** | Reduces loss fluctuation across timesteps |
| 🎨 **Multiple Loss Modes** | Frequency-aware / Style-structure / Unified |
| 🔧 **Auto Hardware Optimization** | Detects GPU and auto-configures (Tier S/A/B) |
| 🖥️ **Modern WebUI** | Vue.js + FastAPI full-stack interface |
| 📊 **Real-time Monitoring** | Loss curves, progress, VRAM monitoring |
| 🏷️ **Ollama Tagging** | One-click AI image captioning |
| 🔄 **Multi-Model Support** | Switch between Z-Image / LongCat-Image |

---

## 🚀 Quick Start

### Step 1: Install PyTorch (Required)

Choose based on your CUDA version:

```bash
# CUDA 12.8 (RTX 40 series recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Flash Attention (Recommended)

Flash Attention significantly reduces VRAM usage and speeds up training.

**Linux** - Download from [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases):

```bash
# Check your environment versions
python --version                                      # e.g.: Python 3.12
python -c "import torch; print(torch.version.cuda)"  # e.g.: 12.8

# Download matching version (example: Python 3.12 + CUDA 12 + PyTorch 2.5)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install
pip install flash_attn-*.whl
```

**Windows** - Download prebuilt from [AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl/tree/main):

```batch
:: Example: Python 3.12 + CUDA 12.8 + PyTorch 2.9.1
pip install https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl
```

> **Tip**: If no matching version exists, skip this step. The program will automatically use SDPA as fallback.

### Step 3: Install Diffusers (Required)

⚠️ **Note**: This project requires diffusers 0.36+ (dev version), install from git:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### Step 4: One-Click Deploy

#### Linux / Mac

```bash
# Clone project
git clone https://github.com/None9527/None_Z-image-Turbo_trainer.git
cd None_Z-image-Turbo_trainer

# One-click install dependencies
chmod +x setup.sh
./setup.sh

# Edit config (set model paths)
cp env.example .env
nano .env

# Start service
./start.sh
```

#### Windows

```batch
:: Clone project
git clone https://github.com/None9527/None_Z-image-Turbo_trainer.git
cd None_Z-image-Turbo_trainer

:: One-click install (double-click or command line)
setup.bat

:: Edit config (set model paths)
copy env.example .env
notepad .env

:: Start service
start.bat
```

### Step 5: Access Web UI

After deployment, open browser: **http://localhost:9198**

---

## 📦 Manual Installation (Optional)

<details>
<summary>Expand for manual installation if one-click deploy fails</summary>

### ⚠️ Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (for frontend build)
- **npm** or **pnpm**

### Installation Steps

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install latest diffusers
pip install git+https://github.com/huggingface/diffusers.git

# 3. Install this project
pip install -e .

# 4. Build frontend (Important!)
cd webui-vue
npm install          # or pnpm install
npm run build        # generates dist directory
cd ..

# 5. Create config file
cp env.example .env

# 6. Start service
cd webui-vue/api && python main.py --port 9198
```

> **💡 Tip**: If `npm run build` fails, ensure Node.js version >= 18.
> Use `node -v` to check version.

</details>

---

## 🖥️ Command Line Usage (Advanced)

Besides Web UI, you can use command line directly:

### Generate Cache

```bash
# Generate Latent cache (VAE encoding)
python -m zimage_trainer.cache_latents \
    --model_path ./zimage_models \
    --dataset_path ./datasets/your_dataset \
    --output_dir ./datasets/your_dataset

# Generate Text cache (text encoding)
python -m zimage_trainer.cache_text_encoder \
    --text_encoder ./zimage_models/text_encoder \
    --input_dir ./datasets/your_dataset \
    --output_dir ./datasets/your_dataset \
    --max_length 512  # Optional: 256/512/1024, default 512
```

### Start Training

First copy example config and modify paths:

```bash
# Z-Image training
cp config/acrf_config.toml config/my_zimage_config.toml
# Edit my_zimage_config.toml, modify [model].dit and [[dataset.sources]].cache_directory

# LongCat-Image training
cp config/longcat_turbo_config.toml config/my_longcat_config.toml
# Edit my_longcat_config.toml, modify [model].dit and [[dataset.sources]].cache_directory
```

Then start training:

```bash
# Z-Image training (recommend using accelerate)
python -m accelerate.commands.launch --mixed_precision bf16 \
    scripts/train_zimage_v2.py --config config/my_zimage_config.toml

# LongCat-Image training
python -m accelerate.commands.launch --mixed_precision bf16 \
    scripts/train_longcat.py --config config/my_longcat_config.toml
```

> **⚠️ Important**: Must modify these paths in config:
> - `[model].dit` - Transformer model path
> - `[model].output_dir` - Output directory
> - `[[dataset.sources]].cache_directory` - Dataset cache path

### Inference

```bash
# Load LoRA and generate image
python -m zimage_trainer.inference \
    --model_path ./zimage_models \
    --lora_path ./output/your_lora.safetensors \
    --prompt "your prompt here" \
    --output_path ./output/generated.png \
    --num_inference_steps 10
```

### Start Web UI Service

```bash
# Method 1: Use script
./start.sh          # Linux/Mac
start.bat           # Windows

# Method 2: Direct start
cd webui-vue/api
python main.py --port 9198 --host 0.0.0.0

# Method 3: Use uvicorn (hot reload)
cd webui-vue/api
uvicorn main:app --port 9198 --reload
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Service config
TRAINER_PORT=9198           # Web UI port
TRAINER_HOST=0.0.0.0        # Listen address

# Model paths
MODEL_PATH=/./zimage_models

# Dataset path
DATASET_PATH=./datasets

# Ollama config
OLLAMA_HOST=http://127.0.0.1:11434
```

### Training Parameters (`config/acrf_config.toml`)

```toml
[acrf]
turbo_steps = 10        # Anchor count (inference steps)
shift = 3.0             # Z-Image official value
jitter_scale = 0.02     # Anchor jitter

[lora]
network_dim = 16        # LoRA rank
network_alpha = 16      # LoRA alpha

[training]
learning_rate = 1e-4    # Learning rate
num_train_epochs = 10   # Training epochs
snr_gamma = 5.0         # Min-SNR weighting
loss_mode = "standard"  # Loss mode (see below)

[dataset]
batch_size = 1
enable_bucket = true
max_sequence_length = 512  # Text sequence length (must match cache)
```

### 🎨 Loss Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **standard** | Basic MSE + optional FFT/Cosine | General training |
| **frequency** | Frequency-aware (HF L1 + LF Cosine) | Sharpen details |
| **style** | Style-structure (SSIM + Lab stats) | Learn lighting/color style |
| **unified** | Frequency + Style combined | Full enhancement |

> 💡 **Beginners**: Start with `standard` mode, try others if unsatisfied.

#### 📐 Freq Sub-parameters

| Parameter | Default | Function | Recommended |
|-----------|---------|----------|-------------|
| `alpha_hf` | 1.0 | High-freq (texture/edge) enhancement | 0.5 ~ 1.5 |
| `beta_lf` | 0.2 | Low-freq (structure/lighting) lock | 0.1 ~ 0.5 |

**Scenario Configs:**

| Scenario | alpha_hf | beta_lf | Notes |
|----------|----------|---------|-------|
| **Sharpen Details** | 1.0~1.5 | 0.1 | Focus on textures |
| **Keep Structure** | 0.5 | 0.3~0.5 | Prevent composition shift |
| **⭐ Balanced** | 0.8 | 0.2 | Recommended default |

#### 🎨 Style Sub-parameters

| Parameter | Default | Function | Recommended |
|-----------|---------|----------|-------------|
| `lambda_struct` | 1.0 | SSIM structure lock (prevent face collapse) | 0.5 ~ 1.5 |
| `lambda_light` | 0.5 | L-channel stats (learn lighting curves) | 0.3 ~ 1.0 |
| `lambda_color` | 0.3 | ab-channel stats (learn color preference) | 0.1 ~ 0.5 |
| `lambda_tex` | 0.5 | High-freq L1 (texture enhancement) | 0.3 ~ 0.8 |

**Scenario Configs:**

| Scenario | struct | light | color | tex | Notes |
|----------|--------|-------|-------|-----|-------|
| **Portrait** | 1.5 | 0.3 | 0.2 | 0.3 | Strong structure lock |
| **Style Transfer** | 0.5 | 0.8 | 0.5 | 0.3 | Focus on lighting/color |
| **Detail Enhancement** | 0.8 | 0.3 | 0.2 | 0.8 | Sharpen textures |
| **⭐ Balanced** | 1.0 | 0.5 | 0.3 | 0.5 | Recommended default |

> ⚠️ **Note**: When both Freq and Style are enabled, high-freq penalties overlap (`alpha_hf` and `lambda_tex`). Consider reducing one.

### Hardware Tiers

| Tier | VRAM | GPU Examples | Auto Optimization |
|------|------|--------------|-------------------|
| **S** | 32GB+ | A100/H100/5090 | Full performance |
| **A** | 24GB | 3090/4090 | High performance, native SDPA |
| **B** | 16GB | 4080/4070Ti | Balanced mode |

---

## 📊 Workflow

| Step | Function | Description |
|:---:|:---:|:---|
| 1️⃣ | **Dataset** | Import images, Ollama AI captioning |
| ➡️ | | |
| 2️⃣ | **Cache** | Pre-compute Latent and Text embeddings |
| ➡️ | | |
| 3️⃣ | **Train** | AC-RF LoRA fine-tuning |
| ➡️ | | |
| 4️⃣ | **Generate** | Load LoRA and test results |

---

## 🔧 FAQ

<details>
<summary><strong>Q: Loss fluctuates a lot (0.08-0.6)?</strong></summary>

A: Normal! Different sigma values have different prediction difficulty. Watch if **EMA loss** trends downward overall.

</details>

<details>
<summary><strong>Q: CUDA Out of Memory?</strong></summary>

A: Try these methods:
- Increase `gradient_accumulation_steps` (e.g., 4 → 8)
- Reduce `network_dim` (e.g., 32 → 16)
- Ensure Flash Attention is installed

</details>

<details>
<summary><strong>Q: How many epochs?</strong></summary>

A: Depends on dataset size:
- < 50 images: 10-15 epochs
- 50-200 images: 8-10 epochs
- \> 200 images: 5-8 epochs

</details>

---

## 📬 Contact

- 📧 lihaonan1082@gmail.com
- 📮 592532681@qq.com

---

## 📝 License

Apache 2.0

## 🙏 Acknowledgements

- [Z-Image](https://github.com/Alpha-VLLM/Lumina-Image) - Base model
- [diffusers](https://github.com/huggingface/diffusers) - Training framework
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Efficient attention
  
---

<div align="center">

**Made with ❤️ by None**

</div>
