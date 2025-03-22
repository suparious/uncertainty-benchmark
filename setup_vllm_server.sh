#!/bin/bash
# Setup a vLLM-OpenAI compatible server for benchmarking

# Check if model name is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_name> [gpu_ids]"
  echo "Example: $0 meta-llama/Llama-2-13b-hf 0,1"
  exit 1
fi

MODEL_NAME=$1
GPU_IDS=${2:-"0"}  # Default to GPU 0 if not specified
PORT=8000

# Create a Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install vLLM
echo "Installing vLLM and dependencies..."
pip install vllm

# Start the vLLM OpenAI-compatible server
echo "Starting vLLM server for model: $MODEL_NAME on GPUs: $GPU_IDS"
echo "Server will be available at http://localhost:$PORT/v1"
echo "Press Ctrl+C to stop the server"

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size $(echo $GPU_IDS | tr ',' '\n' | wc -l) \
  --dtype bfloat16 \
  --gpu $GPU_IDS \
  --port $PORT

# Deactivate the virtual environment when done
deactivate