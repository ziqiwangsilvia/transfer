#!/bin/bash
# End-to-end: for each model, run inference then immediately evaluate.
# vLLM server is started and stopped once per model.

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
BENCHMARKS=""
MAX_SAMPLES=""
BATCH_SIZE=""
POST_PROCESSED=""
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG_FILE="$2"; shift 2 ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --max-samples)     MAX_SAMPLES="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --post-processed) POST_PROCESSED="--post-processed"; shift ;;
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

# vLLM lifecycle

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

# Per-model inference + eval

SUCCESSFUL=0
FAILED=0
FAILED_MODELS=()

run_model() {
    local MODEL="$1"
    local BACKEND="$2"
    local TP_SIZE="${3:-1}"

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Model:   $MODEL${NC}"
    echo -e "${GREEN}Backend: $BACKEND${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    # --- Inference ---
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

    echo -e "${BLUE}[Inference] $INFER_CMD${NC}\n"

    # Run inference with full live output, capture only the last line (output dir)
    local INFER_OUT
    INFER_OUT=$(eval $INFER_CMD | tee /dev/stderr | tail -1)
    local INFER_STATUS=${PIPESTATUS[0]}

    [ "$BACKEND" = "vllm-service" ] && stop_vllm_server

    if [ $INFER_STATUS -ne 0 ]; then
        echo -e "\n${RED}Inference failed for $MODEL${NC}"
        ((FAILED++))
        FAILED_MODELS+=("$MODEL [inference]")
        [ "$CONTINUE_ON_ERROR" = "false" ] && return 1
        return 0
    fi

    echo -e "\n${GREEN}Inference complete → $INFER_OUT${NC}"

    # --- Eval ---
    local EVAL_CMD="python eval.py --inference-dir \"$INFER_OUT\" --config \"$CONFIG_FILE\""
    [ -n "$BENCHMARKS" ]      && EVAL_CMD="$EVAL_CMD --benchmarks $BENCHMARKS"
    [ -n "$POST_PROCESSED" ] && EVAL_CMD="$EVAL_CMD $POST_PROCESSED"

    echo -e "\n${BLUE}[Eval] $EVAL_CMD${NC}\n"

    if eval $EVAL_CMD; then
        echo -e "\n${GREEN}Evaluation complete for $MODEL${NC}"
        ((SUCCESSFUL++))
    else
        echo -e "\n${RED}Evaluation failed for $MODEL${NC}"
        ((FAILED++))
        FAILED_MODELS+=("$MODEL [eval]")
        [ "$CONTINUE_ON_ERROR" = "false" ] && return 1
    fi

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

TOTAL_MODELS=$(( ${#VLLM_SERVICE_MODELS[@]} + ${#OPENAI_MODELS[@]} ))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}End-to-End Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Config:     $CONFIG_FILE"
echo "Models:     $TOTAL_MODELS  (vLLM: ${#VLLM_SERVICE_MODELS[@]}, OpenAI: ${#OPENAI_MODELS[@]})"
echo "Benchmarks: ${BENCHMARKS:-all enabled}"
echo ""

echo -e "${BLUE}Running vLLM Service Models...${NC}\n"
for MODEL_CONFIG in "${VLLM_SERVICE_MODELS[@]}"; do
    IFS=':' read -r MODEL TP_SIZE <<< "$MODEL_CONFIG"
    run_model "$MODEL" "vllm-service" "$TP_SIZE" || break
done

echo -e "\n${BLUE}Running OpenAI Models...${NC}\n"
for MODEL in "${OPENAI_MODELS[@]}"; do
    run_model "$MODEL" "openai" "1" || break
done


# Summary

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed:     $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed models:${NC}"
    for M in "${FAILED_MODELS[@]}"; do echo -e "  - $M"; done
fi

echo -e "\n${BLUE}Next steps:${NC}"
echo "  Single:  python analyse_results.py --results <dir>/<benchmark>/per_item_results.parquet"
echo "  Compare: python analyse_results.py --compare outputs/eval --benchmark tool_calling --output tc.png"
echo "           python analyse_results.py --compare outputs/eval --benchmark guardrailing --output gr.png"
echo "           python analyse_results.py --compare outputs/eval --benchmark content --output content.png"

[ $SUCCESSFUL -eq 0 ] && { echo -e "\n${RED}All models failed.${NC}"; exit 1; }
[ $FAILED -gt 0 ]     && exit 0
echo -e "\n${GREEN}All models completed successfully.${NC}"
exit 0
