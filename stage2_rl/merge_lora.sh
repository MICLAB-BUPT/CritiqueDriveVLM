#!/bin/bash
# ============================================================================
# Merge a verl FSDP RL checkpoint into a standalone HuggingFace model.
# Run this after Stage 2 (run_grpo.sh) to obtain the deployable Teacher model.
# ============================================================================
set -x

VERL_CODE_DIR="/path/to/verl"

# Checkpoint to merge (point at the `actor` sub-dir of a global_step_* dir)
CHECKPOINT_DIR="/path/to/models/rl_checkpoints/global_step_350/actor"

# Output dir for the merged model
TARGET_DIR="/path/to/models/Qwen3-VL-8B-teacher"

export PYTHONPATH="$VERL_CODE_DIR:$PYTHONPATH"
cd "$VERL_CODE_DIR"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CHECKPOINT_DIR" \
    --target_dir "$TARGET_DIR"

echo "Merge done. Model saved to: $TARGET_DIR"
