#!/bin/bash

# Dataset Creation Script
# Loads HuggingFace dataset and formats it according to schema

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Dataset Creation and Formatting Script${NC}"
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

# Check if config.yaml exists
if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
    echo -e "${RED}❌ config.yaml not found in $SCRIPT_DIR${NC}"
    exit 1
fi

echo -e "${YELLOW}📌 Config file: $SCRIPT_DIR/config.yaml${NC}"
echo ""

# Check if required packages are installed
echo -e "${YELLOW}📌 Checking dependencies...${NC}"
python3 -c "import yaml; import datasets" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install pyyaml datasets
}

echo -e "${GREEN}✅ Dependencies ready${NC}"
echo ""

# Run the dataset creation script
echo -e "${YELLOW}🚀 Starting dataset creation...${NC}"
echo ""

cd "$SCRIPT_DIR"
python3 create_dataset.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Dataset creation completed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    # Show output file info if it exists
    OUTPUT_FILE=$(grep "output_path:" "$SCRIPT_DIR/config.yaml" | grep -oE '"[^"]*"' | tr -d '"')
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="formatted_dataset.json"
    fi
    
    if [ -f "$SCRIPT_DIR/$OUTPUT_FILE" ]; then
        SIZE=$(wc -c < "$SCRIPT_DIR/$OUTPUT_FILE")
        LINES=$(wc -l < "$SCRIPT_DIR/$OUTPUT_FILE")
        echo -e "${BLUE}📊 Output file: $OUTPUT_FILE${NC}"
        echo -e "${BLUE}   Size: $(numfmt --to=iec $SIZE 2>/dev/null || echo $SIZE bytes)${NC}"
        echo -e "${BLUE}   Lines: $LINES${NC}"
    fi
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ Dataset creation failed!${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    exit 1
fi
