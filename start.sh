#!/bin/bash
# ============================================================================
# None Trainer - Linux/Mac 启动脚本
# ============================================================================
#
# 使用方法:
#   ./start.sh              # 默认启动
#   ./start.sh --port 8080  # 指定端口
#   ./start.sh --dev        # 开发模式（热重载）
#
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认配置
DEFAULT_PORT=9198
DEFAULT_HOST="0.0.0.0"
DEV_MODE=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            TRAINER_PORT="$2"
            shift 2
            ;;
        --host|-H)
            TRAINER_HOST="$2"
            shift 2
            ;;
        --dev|-d)
            DEV_MODE=1
            shift
            ;;
        --help|-h)
            echo "None Trainer 启动脚本"
            echo ""
            echo "使用方法: ./start.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --port, -p PORT    指定端口 (默认: 9198)"
            echo "  --host, -H HOST    指定监听地址 (默认: 0.0.0.0)"
            echo "  --dev, -d          开发模式（热重载）"
            echo "  --help, -h         显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 加载 .env 配置
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# 应用默认值
export TRAINER_PORT=${TRAINER_PORT:-$DEFAULT_PORT}
export TRAINER_HOST=${TRAINER_HOST:-$DEFAULT_HOST}
export MODEL_PATH=${MODEL_PATH:-"$SCRIPT_DIR/models"}
export DATASET_PATH=${DATASET_PATH:-"$SCRIPT_DIR/datasets"}
export LORA_PATH=${LORA_PATH:-"$SCRIPT_DIR/output"}
export OLLAMA_HOST=${OLLAMA_HOST:-"http://127.0.0.1:11434"}
export OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.2-vision"}

# 确保必要目录存在
mkdir -p "$DATASET_PATH" "$LORA_PATH" logs

# 激活虚拟环境
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# 设置 Python 路径
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Banner
clear
echo -e "${CYAN}"
echo "  _   _                    _____          _                 "
echo " | \ | |                  |_   _|        (_)                "
echo " |  \| | ___  _ __   ___    | |_ __ __ _ _ _ __   ___ _ __  "
echo " | . \` |/ _ \| '_ \ / _ \   | | '__/ _\` | | '_ \ / _ \ '__| "
echo " | |\  | (_) | | | |  __/   | | | | (_| | | | | |  __/ |    "
echo " |_| \_|\___/|_| |_|\___|   \_/_|  \__,_|_|_| |_|\___|_|    "
echo -e "${NC}"
echo ""

# 显示配置
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  服务配置${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  端口:       ${YELLOW}$TRAINER_PORT${NC}"
echo -e "  监听:       ${YELLOW}$TRAINER_HOST${NC}"
echo -e "  模型路径:   ${YELLOW}$MODEL_PATH${NC}"
echo -e "  数据集:     ${YELLOW}$DATASET_PATH${NC}"
echo -e "  LoRA输出:   ${YELLOW}$LORA_PATH${NC}"
echo -e "  Ollama:     ${YELLOW}$OLLAMA_HOST${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检查 Ollama 服务
echo -e "${BLUE}[检查服务]${NC}"
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "  Ollama: ${GREEN}✓ 运行中${NC}"
else
    echo -e "  Ollama: ${YELLOW}✗ 未运行 (图片标注功能不可用)${NC}"
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU: ${GREEN}✓ $GPU_NAME ($GPU_MEM)${NC}"
else
    echo -e "  GPU: ${RED}✗ 未检测到${NC}"
fi
echo ""

# 检查前端是否需要重新构建
DIST_DIR="$SCRIPT_DIR/webui-vue/dist"
SRC_DIR="$SCRIPT_DIR/webui-vue/src"

check_frontend_build() {
    # 如果 dist 不存在，需要构建
    if [ ! -d "$DIST_DIR" ]; then
        return 0
    fi
    
    # 检查 src 目录是否比 dist 新
    NEWEST_SRC=$(find "$SRC_DIR" -type f \( -name "*.vue" -o -name "*.ts" -o -name "*.tsx" \) -newer "$DIST_DIR/index.html" 2>/dev/null | head -1)
    if [ -n "$NEWEST_SRC" ]; then
        return 0
    fi
    
    return 1
}

if check_frontend_build; then
    echo -e "${YELLOW}[前端构建] 检测到代码更新，重新构建...${NC}"
    cd "$SCRIPT_DIR/webui-vue"
    if command -v npm &> /dev/null; then
        npm run build --silent
        echo -e "${GREEN}[前端构建] ✓ 构建完成${NC}"
    else
        echo -e "${RED}[前端构建] ✗ 未找到 npm，请先安装 Node.js${NC}"
        echo -e "${YELLOW}  或手动运行: cd webui-vue && npm run build${NC}"
    fi
    cd "$SCRIPT_DIR"
fi
echo ""

# 获取本机 IP
get_local_ip() {
    # Linux
    if command -v hostname &> /dev/null; then
        hostname -I 2>/dev/null | awk '{print $1}'
    # macOS
    elif command -v ipconfig &> /dev/null; then
        ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null
    else
        echo "127.0.0.1"
    fi
}

LOCAL_IP=$(get_local_ip)

# 启动 TensorBoard
LOG_DIR="$LORA_PATH/logs"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}启动 TensorBoard...${NC}"
# 检查端口是否被占用 (简单检查)
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "  端口 6006 被占用，跳过启动 TensorBoard"
else
    tensorboard --logdir "$LOG_DIR" --port 6006 --host 0.0.0.0 > /dev/null 2>&1 &
    TB_PID=$!
    echo -e "  地址:   ${CYAN}http://localhost:6006${NC}"
    if [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "127.0.0.1" ]; then
        echo -e "  局域网: ${CYAN}http://$LOCAL_IP:6006${NC}"
    fi
fi

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}正在停止服务...${NC}"
    if [ -n "$TB_PID" ]; then
        kill "$TB_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# 启动服务
echo -e "${GREEN}启动 Web UI...${NC}"
echo -e "访问地址:"
echo -e "  本机:   ${CYAN}http://localhost:$TRAINER_PORT${NC}"
if [ "$TRAINER_HOST" = "0.0.0.0" ] && [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "127.0.0.1" ]; then
    echo -e "  局域网: ${CYAN}http://$LOCAL_IP:$TRAINER_PORT${NC}"
fi
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动命令
cd "$SCRIPT_DIR/webui-vue/api"

if [ "$DEV_MODE" -eq 1 ]; then
    echo -e "${YELLOW}[开发模式] 热重载已启用${NC}"
    python -m uvicorn main:app \
        --host "$TRAINER_HOST" \
        --port "$TRAINER_PORT" \
        --reload \
        --reload-dir "$SCRIPT_DIR/webui-vue/api" \
        --log-level info
else
    python -m uvicorn main:app \
        --host "$TRAINER_HOST" \
        --port "$TRAINER_PORT" \
        --log-level warning
fi

