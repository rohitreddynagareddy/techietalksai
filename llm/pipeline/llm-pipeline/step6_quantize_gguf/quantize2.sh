#!/bin/bash
set -e

INPUT_PATH="model-f16.gguf"
OUTPUT_PATH="model-q4_0.gguf"
QUANT_TYPE="q4_0"

# Ensure llama.cpp repo is present (we can copy it from previous stage or clone again)
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    make -C llama.cpp -j $(nproc)  # build the tools (including llama-quantize)
fi

# Run quantization
llama.cpp/llama-quantize "$INPUT_PATH" "$OUTPUT_PATH" "$QUANT_TYPE"

echo "Quantization complete: $OUTPUT_PATH"
