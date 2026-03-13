#!/bin/bash
# Run evaluation on one or more inference output directories.
# Usage:
#   ./run_eval.sh --inference-dir outputs/eval/2026-01-01/model
#   ./run_eval.sh --inference-dirs-file dirs.txt
#   run_inference.sh | grep "^outputs/" | ./run_eval.sh --stdin

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Config resolution: CLI arg > ./config.yaml > config/evaluation_default.yaml
CONFIG_FILE=""
INFERENCE_DIRS=()
DIRS_FILE=""
FROM_STDIN=false
BENCHMARKS=""
POST_PROCESSED=""
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG_FILE="$2"; shift 2 ;;
        --inference-dir)    INFERENCE_DIRS+=("$2"); shift 2 ;;
        --inference-dirs-file) DIRS_FILE="$2"; shift 2 ;;
        --stdin)            FROM_STDIN=true; shift ;;
        --benchmarks)       BENCHMARKS="$2"; shift 2 ;;
        --post-processed)  POST_PROCESSED="--post-processed $2"; shift 2 ;;
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

# Collect inference dirs from file
if [ -n "$DIRS_FILE" ]; then
    if [ ! -f "$DIRS_FILE" ]; then
        echo -e "${RED}Dirs file not found: $DIRS_FILE${NC}"
        exit 1
    fi
    while IFS= read -r line; do
        [ -n "$line" ] && INFERENCE_DIRS+=("$line")
    done < "$DIRS_FILE"
fi

# Collect inference dirs from stdin
if [ "$FROM_STDIN" = true ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && INFERENCE_DIRS+=("$line")
    done
fi

if [ ${#INFERENCE_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}No inference directories provided.${NC}"
    echo "Usage:"
    echo "  $0 --inference-dir <dir>"
    echo "  $0 --inference-dirs-file <file>"
    echo "  run_inference.sh | grep '^outputs/' | $0 --stdin"
    exit 1
fi

# Evaluation runner
SUCCESSFUL=0
FAILED=0
FAILED_DIRS=()

run_eval() {
    local INFER_DIR="$1"

    if [ ! -d "$INFER_DIR" ]; then
        echo -e "${RED}Directory not found: $INFER_DIR${NC}"
        ((FAILED++))
        FAILED_DIRS+=("$INFER_DIR (not found)")
        return 0
    fi

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Evaluating: $INFER_DIR${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    local EVAL_CMD="python eval.py --inference-dir \"$INFER_DIR\" --config \"$CONFIG_FILE\""
    [ -n "$BENCHMARKS" ]    && EVAL_CMD="$EVAL_CMD --benchmarks $BENCHMARKS"
    [ -n "$POST_PROCESSED" ] && EVAL_CMD="$EVAL_CMD $POST_PROCESSED"

    echo "Command: $EVAL_CMD"
    echo ""

    if eval $EVAL_CMD; then
        echo -e "\n${GREEN}Evaluation complete: $INFER_DIR${NC}"
        ((SUCCESSFUL++))
    else
        echo -e "\n${RED}Evaluation failed: $INFER_DIR${NC}"
        ((FAILED++))
        FAILED_DIRS+=("$INFER_DIR")
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation — ${#INFERENCE_DIRS[@]} director$([ ${#INFERENCE_DIRS[@]} -eq 1 ] && echo y || echo ies)${NC}"
echo -e "${BLUE}========================================${NC}\n"

for DIR in "${INFERENCE_DIRS[@]}"; do
    run_eval "$DIR" || break
done

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed:     $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed directories:${NC}"
    for D in "${FAILED_DIRS[@]}"; do echo -e "  - $D"; done
fi

echo -e "\n${BLUE}Next steps:${NC}"
echo "  Single:  python analyse_results.py --results <dir>/<benchmark>/per_item_results.parquet"
echo "  Compare: python analyse_results.py --compare outputs/eval --benchmark tool_calling --output tc.png"
echo "           python analyse_results.py --compare outputs/eval --benchmark guardrailing --output gr.png"
echo "           python analyse_results.py --compare outputs/eval --benchmark content --output content.png"

[ $SUCCESSFUL -eq 0 ] && { echo -e "\n${RED}All evaluations failed.${NC}"; exit 1; }
[ $FAILED -gt 0 ]     && exit 0
echo -e "\n${GREEN}All evaluations completed successfully.${NC}"
exit 0
