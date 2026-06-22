#!/bin/bash

# Define the absolute path to the project directory
PROJECT_DIR="/fsx/users/wangzg/prompt-distillation"

# Change to the project directory. Exit if it fails.
cd "$PROJECT_DIR" || { echo "ERROR: Failed to change directory to $PROJECT_DIR" >&2; exit 1; }
echo "Successfully changed working directory to: $PWD"

unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH has been unset"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ====================================================================
# Configuration
# ====================================================================
# Pipeline configs to run. Each gets its own full pipeline execution.
# These paths are relative to PROJECT_DIR on the cluster.
CONFIGS=(
    # Baseline: lora_r=256, token=1.0, logit=0.5
    "config/experiments_20260416/test_mode.yaml"

    # Axis 1: LoRA rank sweep â€” does smaller adapter preserve base model ability?
    # "config/pipeline_lora_r128.yaml"
    # "config/pipeline_lora_r64.yaml"

    # # Axis 2: Token loss weight â€” does stronger direct supervision on NLP text help?
    # "config/pipeline_tok2_log05.yaml"       # token=2.0, logit=0.5
    # "config/pipeline_tok1_log0.yaml"        # token=1.0, logit=0.0 (pure SFT, no KD)
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Prompt Distillation - Multi-Config Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Number of configs: ${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $cfg"
done
echo ""

# Check if venv activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Warning: Virtual environment not activated${NC}"
    echo "Attempting to activate .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "Venv activation script sourced."
    else
        echo -e "${RED}Error: .venv not found. Please run setup.sh first.${NC}"
        exit 1
    fi
fi

# ====================================================================
# Python environment diagnostics
# ====================================================================
echo ""
echo -e "${YELLOW}--- PYTHON ENVIRONMENT DIAGNOSTICS ---${NC}"
echo "VIRTUAL_ENV variable is set to: $VIRTUAL_ENV"
echo "The 'python' command resolves to: $(which python)"
echo ""
echo "--- Probing the Python interpreter directly ---"
"$(which python)" -c "import sys; print('Python version:', sys.version); print('--- sys.path ---'); [print(p) for p in sys.path];"
echo -e "${YELLOW}--- END OF DIAGNOSTICS ---${NC}"
echo ""

# Track results
SUCCESSFUL=0
FAILED=0
FAILED_CONFIGS=()

run_pipeline() {
    local CONFIG="$1"

    # Extract run_name and key info from config for display
    local RUN_INFO
    RUN_INFO=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
p = c.get('project', {})
m = c.get('models', {})
d = c.get('dataset', {})
g = c.get('gpu', {})
print(f'run_name={p.get(\"run_name\", \"unknown\")}')
print(f'teacher={m.get(\"teacher\", \"unknown\")}')
print(f'student={m.get(\"student\", \"unknown\")}')
print(f'dataset={d.get(\"family\", \"\")}/{d.get(\"name\", \"\")}')
print(f'train_only={p.get(\"train_only\", False)}')
print(f'vllm_gpu={g.get(\"vllm\", \"0\")}')
print(f'train_gpu={g.get(\"train\", \"0\")}')
")

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Running pipeline: $CONFIG${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "$RUN_INFO" | while IFS= read -r line; do
        echo -e "  ${BLUE}${line}${NC}"
    done
    echo ""

    if bash scripts/run_pipeline.sh "$CONFIG"; then
        echo -e "\n${GREEN}Successfully completed: $CONFIG${NC}"
        ((SUCCESSFUL++))
    else
        echo -e "\n${RED}Failed: $CONFIG${NC}"
        ((FAILED++))
        FAILED_CONFIGS+=("$CONFIG")
    fi
}

# Run each config
for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo -e "${RED}Config not found: $CONFIG -- skipping${NC}"
        ((FAILED++))
        FAILED_CONFIGS+=("$CONFIG (not found)")
        continue
    fi
    run_pipeline "$CONFIG"
done

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total configs: ${#CONFIGS[@]}"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed configs:${NC}"
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo -e "  - $cfg"
    done
fi

if [ $SUCCESSFUL -eq 0 ]; then
    echo -e "\n${RED}All configs failed.${NC}"
    exit 1
elif [ $FAILED -gt 0 ]; then
    echo -e "\n${YELLOW}Completed with some failures.${NC}"
    exit 0
else
    echo -e "\n${GREEN}All configs completed successfully.${NC}"
    exit 0
fi
