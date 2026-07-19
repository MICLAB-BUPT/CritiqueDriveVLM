#!/bin/bash
# ============================================================================
# Serve the frozen multi-dimensional Verifier (Stage 1 output) via vLLM.
# The Stage-2 reward function and multi-turn interaction talk to this server
# at http://localhost:8000/v1/chat/completions.
#
# The served-model-name MUST stay "verifier" to match the client
# (reward/verifier_client.py, interaction/reflexion_interaction.py).
# ============================================================================
export CUDA_VISIBLE_DEVICES=5

# Path to the trained Verifier (Qwen3-VL-8B fine-tuned as a safety coach/auditor).
export MODEL_PATH="/path/to/models/Qwen3-VL-8B-verifier"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "verifier" \
    --trust-remote-code \
    --max-num-seqs 256 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --port 8000
