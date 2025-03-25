#!/bin/bash
# Requires docker and a NVIDIA GPU with at least 12GB of memory
export MODEL_ID="solidrust/Hermes-3-Llama-3.1-8B-AWQ"
export MODEL_LENGTH=4096
export GPU_MEMORY_UTILIZATION=0.90
export HF_TOKEN=${HF_TOKEN}
export IMAGE_TAG="latest"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_PORT=8081

docker pull vllm/vllm-openai:${IMAGE_TAG}
docker run --privileged --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}" \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    -p ${VLLM_PORT}:8000 \
    --ipc=host \
    vllm/vllm-openai:${IMAGE_TAG} \
    --model ${MODEL_ID} \
    --tokenizer ${MODEL_ID} \
    --trust-remote-code \
    --dtype auto \
    --device auto \
    --max-model-len ${MODEL_LENGTH}
