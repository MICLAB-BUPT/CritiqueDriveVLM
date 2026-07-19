# Stage 2 — Critique-Driven Multi-Turn RL (Teacher)

GRPO training (via [verl](https://github.com/volcengine/verl)) of the Teacher
policy, guided by the frozen multi-dimensional verifier from Stage 1. The
verifier provides both a **scalar reward** and a **natural-language critique**;
the policy refines its answer over a short multi-turn loop (max `K = 2`), and a
**step-decay penalty** pushes it toward first-attempt correctness.

```
reward/
  drivelmm_reward.py        # verl custom_reward_function (compute_score)
  verifier_client.py        # async client to the frozen verifier (vLLM server)
interaction/
  reflexion_interaction.py  # verl BaseInteraction: multi-turn critique loop
  interaction.yaml          # interaction config (max_attempts: 2)
serve_verifier.sh           # serve the frozen verifier via vLLM
run_grpo.sh                 # launch GRPO training
merge_lora.sh               # merge the verl FSDP checkpoint -> HF model
```

## Reward

Composite reward on the final turn (`reward/drivelmm_reward.py`):

```
R = W_FORMAT                                   # format compliance (0.1, hard gate)
  + W_ACCURACY * R_acc                         # MCQ answer correctness (1.0)
  + W_VERIFIER * (s_per + s_log + s_saf)/3     # process verifier (0.5)
  - (attempts - 1) * DECAY_PER_ATTEMPT         # multi-turn penalty (0.2 / turn)
```

## Run

```bash
# Terminal A: serve the frozen verifier (must stay running during training)
bash serve_verifier.sh

# Terminal B: GRPO training
#   - set VERL_CODE_DIR / MODEL_PATH / DATA_DIR / CKPT_DIR inside the script
bash run_grpo.sh

# After training: merge the checkpoint into a deployable Teacher model
bash merge_lora.sh
```

> The verifier endpoint defaults to `http://localhost:8000/v1/chat/completions`
> and can be overridden with the `VERIFIER_API_URL` environment variable.
> The served model name must stay `verifier` to match the client.

Requires **verl** (installed separately) with multi-turn interaction + vLLM
rollout support.
