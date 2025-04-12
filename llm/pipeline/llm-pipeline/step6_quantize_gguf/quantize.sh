#!/bin/bash
set -e

INPUT_PATH="model-f16.gguf"
OUTPUT_PATH="model-q4_0.gguf"
QUANT_TYPE="q4_0"

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi

# Build llama.cpp using CMake
cd llama.cpp
mkdir -p build
cd build
cmake ..
cmake --build . --parallel
cd ../..

# Run quantization
./llama.cpp/build/bin/llama-quantize "$INPUT_PATH" "$OUTPUT_PATH" "$QUANT_TYPE"

echo "✅ Quantization complete: $OUTPUT_PATH"
