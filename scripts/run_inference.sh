#!/bin/bash
# Run inference for each model, starting/stopping a vLLM server as needed.
# Prints each output directory to stdout on completion — pipe into run_eval.sh.

unset LD_LIBRARY_PATH

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Models
VLLM_SERVICE_MODELS=(
    "RedHatAI/gemma-3-12b-it-FP8-dynamic:1"
    # "RedHatAI/gemma-3-27b-it-FP8-dynamic:1"
    # "google/gemma-3-12b-it:1"
    # "google/gemma-3-27b-it:1"
    # "gemma-3-12b-it-gguf/gemma-3-12b-it-q4_0.gguf:1"
    # "gemma-3-27b-it-gguf/gemma-3-27b-it-q4_0.gguf:1"
)

OPENAI_MODELS=(
    # "gpt-4o-mini"
    # "gpt-4o"
)

# Config resolution: CLI arg > ./config.yaml > config/evaluation_default.yaml
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo -e "${RED}Unknown argument: $1${NC}"; exit 1 ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    if [ -f "config.yaml" ]; then
        CONFIG_FILE="config.yaml"
    elif [ -f "config/evaluation_default.yaml" ]; then
        CONFIG_FILE="config/evaluation_default.yaml"
    else
        echo -e "${RED}No config file found. Pass --config or add config.yaml${NC}"
        exit 1
    fi
fi
echo "Using config: $CONFIG_FILE"

MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
BENCHMARKS="${BENCHMARKS:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

# vLLM config
TEMPLATE_HAS_TOOL_TOKEN="${TEMPLATE_HAS_TOOL_TOKEN:-$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG_FILE'))
    print(str(cfg.get('model', {}).get('template_has_tool_token', False)).lower())
except Exception:
    print('false')
" 2>/dev/null)}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG_FILE'))
    print(str(cfg.get('model', {}).get('chat_template', False)).lower())
except Exception:
    print('false'):
" 2>/dev/null)}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"
VLLM_SERVER_LOG="${VLLM_SERVER_LOG:-/tmp/vllm_server.log}"
VLLM_TMUX_WINDOW="vllm_server"

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is required but not found.${NC}"
    exit 1
fi

TMUX_SESSION="${TMUX_SESSION:-$(tmux display-message -p '#S' 2>/dev/null)}"
if [ -z "$TMUX_SESSION" ]; then
    echo -e "${RED}Error: not inside a tmux session. Please run this script from tmux.${NC}"
    exit 1
fi

# vLLM

start_vllm_server() {
    local MODEL="$1"
    local TP_SIZE="${2:-1}"

    echo -e "${BLUE}Launching vLLM server for: $MODEL  (tensor-parallel=$TP_SIZE)${NC}"

    tmux kill-window -t "${TMUX_SESSION}:${VLLM_TMUX_WINDOW}" 2>/dev/null || true

    local TOOL_FLAGS=""
    if [ "$TEMPLATE_HAS_TOOL_TOKEN" = "true" ]; then
        TOOL_FLAGS=" \\
  --enable-auto-tool-choice \\
  --tool-call-parser pythonic"
    if [ "$CHAT_TEMPLATE" = "true"]; then
        TOOL_FLAGS="$TOOL_FLAGS \\
  --chat-template prompts/tool_chat_template_gemma3_pythonic.jinja"
        fi
    fi

    local TOKENIZER_ARG=""
    case "$(basename "$MODEL")" in
        gemma-3-12b-it-q4_0.gguf)
            TOKENIZER_ARG="--tokenizer google/gemma-3-12b-it"
            echo -e "${YELLOW}GGUF detected → tokenizer=google/gemma-3-12b-it${NC}"
            ;;
        gemma-3-27b-it-q4_0.gguf)
            TOKENIZER_ARG="--tokenizer google/gemma-3-27b-it"
            echo -e "${YELLOW}GGUF detected → tokenizer=google/gemma-3-27b-it${NC}"
            ;;
        *)
            echo -e "${GREEN}HF model detected — no tokenizer override${NC}"
            ;;
    esac

    local SERVE_CMD="vllm serve \"$MODEL\" \
  --host $VLLM_HOST \
  --port $VLLM_PORT \
  --tensor-parallel-size $TP_SIZE \
  $TOKENIZER_ARG \
  --enforce-eager \
  --gpu-memory-utilization 0.8${TOOL_FLAGS} \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 8192 \
  --async-scheduling \
  --max-cudagraph-capture-size 2048 \
  --max-num-batched-tokens 8192 \
  --stream-interval 20 \
  2>&1 | tee \"$VLLM_SERVER_LOG\""

    tmux new-window -t "$TMUX_SESSION" -n "$VLLM_TMUX_WINDOW" \
        "bash -c '$SERVE_CMD; echo \"[vllm serve exited with code \$?]\"; read'"

    echo -e "${YELLOW}Waiting for server (timeout: ${VLLM_STARTUP_TIMEOUT}s) — watch: tmux select-window -t ${TMUX_SESSION}:${VLLM_TMUX_WINDOW}${NC}"

    local elapsed=0
    local interval=5
    while [ $elapsed -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if curl -sf "http://$VLLM_HOST:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo -e "${GREEN}vLLM server ready after ${elapsed}s${NC}"
            return 0
        fi
        if ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "$VLLM_TMUX_WINDOW"; then
            echo -e "${RED}vLLM server window closed unexpectedly. Check: $VLLM_SERVER_LOG${NC}"
            tail -20 "$VLLM_SERVER_LOG" 2>/dev/null
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        echo "  Still waiting... (${elapsed}s)"
    done

    echo -e "${RED}Timed out after ${VLLM_STARTUP_TIMEOUT}s${NC}"
    tail -20 "$VLLM_SERVER_LOG" 2>/dev/null
    stop_vllm_server
    return 1
}

stop_vllm_server() {
    if tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "$VLLM_TMUX_WINDOW"; then
        echo -e "${YELLOW}Stopping vLLM server...${NC}"
        tmux kill-window -t "${TMUX_SESSION}:${VLLM_TMUX_WINDOW}"
        echo -e "${GREEN}vLLM server stopped.${NC}"
    fi
}

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    stop_vllm_server
}
trap cleanup EXIT INT TERM

# Inference runner

OUTPUT_DIRS=()
SUCCESSFUL=0
FAILED=0
FAILED_MODELS=()

run_inference() {
    local MODEL="$1"
    local BACKEND="$2"
    local TP_SIZE="${3:-1}"

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Inference: $MODEL${NC}"
    echo -e "${GREEN}Backend:   $BACKEND${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    if [ "$BACKEND" = "vllm-service" ]; then
        if ! start_vllm_server "$MODEL" "$TP_SIZE"; then
            echo -e "${RED}Failed to start vLLM server for $MODEL${NC}"
            ((FAILED++))
            FAILED_MODELS+=("$MODEL (server start failed)")
            [ "$CONTINUE_ON_ERROR" = "false" ] && return 1
            return 0
        fi
    fi

    local INFER_CMD="python infer.py --config \"$CONFIG_FILE\" --model \"$MODEL\" --backend \"$BACKEND\""
    [ -n "$BENCHMARKS" ]  && INFER_CMD="$INFER_CMD --benchmarks $BENCHMARKS"
    [ -n "$MAX_SAMPLES" ] && INFER_CMD="$INFER_CMD --max-samples $MAX_SAMPLES"
    [ -n "$BATCH_SIZE" ]  && INFER_CMD="$INFER_CMD --batch-size $BATCH_SIZE"
    [ "$BACKEND" = "vllm-service" ] && INFER_CMD="$INFER_CMD --api-base http://$VLLM_HOST:$VLLM_PORT/v1"

    echo "Command: $INFER_CMD"

    local INFER_OUT
    INFER_OUT=$(eval $INFER_CMD 2>&1 | tee /dev/stderr | tail -1)
    local INFER_STATUS=${PIPESTATUS[0]}

    [ "$BACKEND" = "vllm-service" ] && stop_vllm_server

    if [ $INFER_STATUS -ne 0 ]; then
        echo -e "\n${RED}Inference failed for $MODEL ($BACKEND)${NC}"
        ((FAILED++))
        FAILED_MODELS+=("$MODEL ($BACKEND)")
        [ "$CONTINUE_ON_ERROR" = "false" ] && return 1
        return 0
    fi

    echo -e "${GREEN}Output dir: $INFER_OUT${NC}"
    OUTPUT_DIRS+=("$INFER_OUT")
    ((SUCCESSFUL++))
    return 0
}

# Main

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}Error: .venv not found. Run setup.sh first.${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}Running vLLM Service Models...${NC}\n"
for MODEL_CONFIG in "${VLLM_SERVICE_MODELS[@]}"; do
    IFS=':' read -r MODEL TP_SIZE <<< "$MODEL_CONFIG"
    run_inference "$MODEL" "vllm-service" "$TP_SIZE" || break
done

echo -e "\n${BLUE}Running OpenAI Models...${NC}\n"
for MODEL in "${OPENAI_MODELS[@]}"; do
    run_inference "$MODEL" "openai" "1" || break
done

# Summary + emit output dirs

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Inference Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed:     $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed models:${NC}"
    for M in "${FAILED_MODELS[@]}"; do echo -e "  - $M"; done
fi

# Print output dirs one per line so run_eval.sh or run_e2e.sh can consume them
if [ ${#OUTPUT_DIRS[@]} -gt 0 ]; then
    echo -e "\n${BLUE}Output directories:${NC}"
    for DIR in "${OUTPUT_DIRS[@]}"; do
        echo "$DIR"
    done
fi

[ $SUCCESSFUL -eq 0 ] && { echo -e "\n${RED}All models failed.${NC}"; exit 1; }
[ $FAILED -gt 0 ]     && exit 0
echo -e "\n${GREEN}All models completed successfully.${NC}"
exit 0
