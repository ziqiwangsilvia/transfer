#!/bin/bash

# API-Bank Dataset Processing Script
# Loads and formats HuggingFace API-Bank dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SRC_DIR="$PROJECT_ROOT/src/benchmark_subset"
CONFIG_DIR="$PROJECT_ROOT/config"
DATASET_DIR="$PROJECT_ROOT/dataset"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  API-Bank Dataset Processing${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}📌 Python version:${NC}"
python3 --version
echo ""

# Accept optional config path as first argument, else use default
# Parse args: accept -c/--config or a positional config path
CONFIG_FILE="$CONFIG_DIR/API-Bank.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            if [[ -n "$2" ]]; then
                CONFIG_FILE="$2"
                shift 2
            else
                echo -e "${RED}❌ Missing value for $1${NC}"
                echo -e "Usage: $0 [-c config.yaml]"
                exit 1
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [-c config.yaml]"
            exit 0
            ;;
        *)
            # If positional looks like a file, treat as config path
            if [[ -f "$1" ]]; then
                CONFIG_FILE="$1"
                shift
            else
                echo -e "${RED}❌ Unknown argument: $1${NC}"
                echo "Usage: $0 [-c config.yaml]"
                exit 1
            fi
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
        echo -e "Usage: $0 [-c config.yaml]"
        exit 1
fi

echo -e "${YELLOW}📌 Config file: $CONFIG_FILE${NC}"
echo -e "${YELLOW}📌 Source directory: $SRC_DIR${NC}"
echo -e "${YELLOW}📌 Output directory: $DATASET_DIR${NC}"
echo ""

# Check if required packages are installed
echo -e "${YELLOW}📌 Checking dependencies...${NC}"
python3 -c "import yaml; import datasets" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install pyyaml datasets
}

echo -e "${GREEN}✅ Dependencies ready${NC}"
echo ""

# Ensure dataset directory exists
mkdir -p "$DATASET_DIR/tool_calling"

# Run the dataset creation script
echo -e "${YELLOW}🚀 Starting API-Bank dataset processing...${NC}"
echo ""

cd "$PROJECT_ROOT"
python3 "$SRC_DIR/create_subset.py"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Dataset processing completed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    # Show output file info if it exists
    OUTPUT_FILE="$DATASET_DIR/tool_calling/API-Bank_subset.json"
    
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(wc -c < "$OUTPUT_FILE")
        LINES=$(wc -l < "$OUTPUT_FILE")
        echo -e "${BLUE}📊 Output file: $OUTPUT_FILE${NC}"
        echo -e "${BLUE}   Size: $(numfmt --to=iec $SIZE 2>/dev/null || echo $SIZE bytes)${NC}"
        echo -e "${BLUE}   Lines: $LINES${NC}"
    fi
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ Dataset processing failed!${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    exit 1
fi
