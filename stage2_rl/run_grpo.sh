#!/bin/bash
# ============================================================================
# CritiqueDriveVLM - Stage 2: Critique-Driven Multi-Turn RL (GRPO via verl)
#
# Launches GRPO training of the Teacher policy with:
#   - a custom multi-dimensional verifier reward  (reward/drivelmm_reward.py)
#   - a multi-turn critique interaction loop       (interaction/interaction.yaml)
#
# Prerequisites:
#   1. Install verl (https://github.com/volcengine/verl) and set VERL_CODE_DIR.
#   2. Serve the frozen verifier first:  bash serve_verifier.sh
#      (the reward fn / interaction talk to it at http://localhost:8000)
#   3. Prepare the RL data as parquet (train/val) with columns:
#      images, question, ground_truth, ...  (see ../README.md).
# ============================================================================
set -x

# ================= 1. Paths (EDIT THESE) =================
VERL_CODE_DIR="/path/to/verl"                        # cloned verl repo
DATA_DIR="/path/to/data/drivelmm_rl"                 # dir with train/val parquet
MODEL_PATH="/path/to/models/Qwen3-VL-8B-sft"         # warm-up SFT checkpoint (Stage 1)
CKPT_DIR="/path/to/models/rl_checkpoints"            # where RL checkpoints are saved

# This repo's stage2_rl dir must be importable so `reward` and `interaction`
# packages resolve.
CODE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REWARD_SCRIPT_PATH="$CODE_ROOT/reward/drivelmm_reward.py"
INTERACTION_CONFIG="$CODE_ROOT/interaction/interaction.yaml"

export PYTHONPATH="$VERL_CODE_DIR:$CODE_ROOT:$PYTHONPATH"

# Ray temp dir
export RAY_TEMP_DIR_PATH="/tmp/ray_tmp"
mkdir -p "$RAY_TEMP_DIR_PATH"
export RAY_TMPDIR="$RAY_TEMP_DIR_PATH"

# (Optional) experiment logging via SwanLab
export SWANLAB_PROJECT="CritiqueDriveVLM-RL"
export SWANLAB_RUN_NAME="Qwen3-VL-8B-teacher-grpo"
export SWANLAB_MODE="cloud"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ================= 2. Launch =================
cd "$VERL_CODE_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_reflexion.parquet \
    data.val_files=$DATA_DIR/test_reflexion.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.image_key=images \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$INTERACTION_CONFIG \
    custom_reward_function.path=$REWARD_SCRIPT_PATH \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=$SWANLAB_PROJECT \
    trainer.experiment_name=$SWANLAB_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 \
    trainer.resume_mode='auto'
