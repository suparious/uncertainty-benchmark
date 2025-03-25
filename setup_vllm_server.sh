#!/bin/bash
# Setup an optimized vLLM-OpenAI compatible server for LLM uncertainty benchmarking
set -e  # Exit on error

# Display help information
function show_help {
  echo "Usage: $0 [OPTIONS] <model_name>"
  echo ""
  echo "Setup an optimized vLLM server with OpenAI-compatible API for LLM benchmarking."
  echo ""
  echo "Options:"
  echo "  -g, --gpus GPUS            GPU IDs to use (comma-separated, default: 0)"
  echo "  -p, --port PORT            Port to serve the API (default: 8000)"
  echo "  -m, --max-model-len LEN    Maximum sequence length (default: 2048)"
  echo "  -b, --batch-size SIZE      Batch size for inference (default: 32)"
  echo "  -w, --workers NUM          Number of worker processes (default: 1)"
  echo "  -q, --quantize TYPE        Quantization type (awq, gptq, none; default: none)"
  echo "  -c, --enable-chat          Enable chat completions endpoint"
  echo "  -t, --trust-remote-code    Trust remote code when loading models"
  echo "  -u, --gpu-util FRACTION    GPU memory utilization (0.0-1.0, default: 0.9)"
  echo "  -s, --swap-space SIZE      CPU swap space in GiB (default: 4)"
  echo "  -d, --dtype TYPE           Model data type (float16, bfloat16, auto, default: bfloat16)"
  echo "  -l, --logprobs NUM         Return top N logprobs for token predictions (default: 10)"
  echo "  -e, --env-path PATH        Path to virtual environment (default: ./venv)"
  echo "  -v, --device DEVICE        Device to use (auto, cuda, cpu, default: auto)"
  echo "  -o, --tool-parser TYPE     Tool call parser type (hermes, llama3_json, mistral, etc.)"
  echo "  -x, --extra-args ARGS      Additional arguments to pass to vLLM (quote the whole string)"
  echo "  -k, --hf-token TOKEN       Hugging Face API token for private models"
  echo "  -h, --help                 Display this help message"
  echo ""
  echo "Example: $0 -g 0,1 -p 8080 -q awq -m 4096 -t -o hermes meta-llama/Llama-3-8b-instruct"
  exit 0
}

# Default values
GPU_IDS="0"
PORT=8000
MAX_MODEL_LEN=2048
BATCH_SIZE=32
WORKERS=1
QUANTIZE="none"
ENABLE_CHAT=false
TRUST_REMOTE_CODE=false
GPU_UTIL=0.9
SWAP_SPACE=4
DTYPE="bfloat16"
LOGPROBS=10
ENV_PATH="./venv"
DEVICE="auto"
TOOL_PARSER=""
EXTRA_ARGS=""
HF_TOKEN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -m|--max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    -b|--batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -w|--workers)
      WORKERS="$2"
      shift 2
      ;;
    -q|--quantize)
      QUANTIZE="$2"
      shift 2
      ;;
    -c|--enable-chat)
      ENABLE_CHAT=true
      shift
      ;;
    -t|--trust-remote-code)
      TRUST_REMOTE_CODE=true
      shift
      ;;
    -u|--gpu-util)
      GPU_UTIL="$2"
      shift 2
      ;;
    -s|--swap-space)
      SWAP_SPACE="$2"
      shift 2
      ;;
    -d|--dtype)
      DTYPE="$2"
      shift 2
      ;;
    -l|--logprobs)
      LOGPROBS="$2"
      shift 2
      ;;
    -e|--env-path)
      ENV_PATH="$2"
      shift 2
      ;;
    -v|--device)
      DEVICE="$2"
      shift 2
      ;;
    -o|--tool-parser)
      TOOL_PARSER="$2"
      shift 2
      ;;
    -x|--extra-args)
      EXTRA_ARGS="$EXTRA_ARGS $2"
      shift 2
      ;;
    -k|--hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information."
      exit 1
      ;;
    *)
      if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="$1"
      else
        echo "Unexpected argument: $1"
        echo "Run '$0 --help' for usage information."
        exit 1
      fi
      shift
      ;;
  esac
done

# Check if model name is provided
if [[ -z "$MODEL_NAME" ]]; then
  echo "Error: Model name is required."
  echo "Run '$0 --help' for usage information."
  exit 1
fi

# Calculate number of GPUs for tensor parallelism
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

# Build arguments
if [[ "$QUANTIZE" != "none" ]]; then
  EXTRA_ARGS+=" --quantization $QUANTIZE"
fi

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  EXTRA_ARGS+=" --trust-remote-code"
fi

if [[ "$ENABLE_CHAT" == "true" ]]; then
  EXTRA_ARGS+=" --chat-template chatml"
fi

if [[ ! -z "$TOOL_PARSER" ]]; then
  EXTRA_ARGS+=" --tool-call-parser $TOOL_PARSER"
fi

# Print configuration
echo "=============================================="
echo "vLLM Server Configuration"
echo "=============================================="
echo "Model:                  $MODEL_NAME"
echo "GPUs:                   $GPU_IDS ($NUM_GPUS GPUs)"
echo "Port:                   $PORT"
echo "Max sequence length:    $MAX_MODEL_LEN"
echo "Batch size:             $BATCH_SIZE"
echo "Workers:                $WORKERS"
echo "Quantization:           $QUANTIZE"
echo "Data type:              $DTYPE"
echo "Device:                 $DEVICE"
echo "Chat mode:              $ENABLE_CHAT"
echo "Trust remote code:      $TRUST_REMOTE_CODE"
echo "Tool call parser:       ${TOOL_PARSER:-none}"
echo "GPU memory utilization: $GPU_UTIL"
echo "CPU swap space:         $SWAP_SPACE GiB"
echo "LogProbs:               $LOGPROBS"
echo "Virtual env path:       $ENV_PATH"
echo "Extra args:             ${EXTRA_ARGS:-none}"
echo "=============================================="

# Create a Python virtual environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
  echo "Creating Python virtual environment at $ENV_PATH..."
  python -m venv "$ENV_PATH"
fi

# Activate the virtual environment
source "$ENV_PATH/bin/activate"

# Install vLLM
echo "Installing vLLM and dependencies..."
pip install --upgrade pip
pip install "vllm>=0.3.0" "fschat>=0.2.30"

# Configure CUDA for better performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1  # Try disabling NCCL P2P for better multi-GPU performance
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Set Hugging Face token if provided
if [[ ! -z "$HF_TOKEN" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "Hugging Face token set for accessing private models"
fi

# Start the vLLM OpenAI-compatible server
echo ""
echo "Starting optimized vLLM server for model: $MODEL_NAME"
echo "Server will be available at http://localhost:$PORT/v1"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server with optimized settings
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --gpu-memory-utilization $GPU_UTIL \
  --tensor-parallel-size $NUM_GPUS \
  --dtype $DTYPE \
  --device $DEVICE \
  --max-model-len $MAX_MODEL_LEN \
  --served-model-name "$(basename $MODEL_NAME)" \
  --port $PORT \
  --distributed-executor-backend ray \
  --max-logprobs $LOGPROBS \
  --swap-space $SWAP_SPACE \
  --max-num-batched-tokens $((MAX_MODEL_LEN * BATCH_SIZE)) \
  --max-num-seqs $BATCH_SIZE \
  --enforce-eager \
  --disable-log-stats \
  $EXTRA_ARGS

# Deactivate the virtual environment when done
deactivate
