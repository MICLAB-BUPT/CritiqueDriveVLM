# Stage 1 — Warm-up SFT & Verifier Construction

This stage produces two models via **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** SFT:

1. **Warm-up policy** — the base VLM fine-tuned to enforce the
   `<think>...</think><answer>...</answer>` reasoning format. This is the
   starting point for Stage-2 RL.
2. **Multi-dimensional verifier** — a frozen judge that scores a response over
   **perception / logic / safety** and produces a critique. Used in Stage 2.

> **This stage is about training.** The datasets are released on HuggingFace —
> you do **not** need to run any data-construction code. Download the data,
> register it, and run the two SFT configs below.

```
stage1_verifier/
├── configs/
│   ├── warmup_sft.yaml            # LoRA SFT of the base policy (<think>/<answer>)
│   ├── verifier_sft.yaml          # LoRA SFT of the verifier (scores + critique)
│   └── dataset_info_snippet.json  # entries to add to LLaMA-Factory's dataset_info.json
├── preprocess/
│   └── stitch_multiview.py        # stitch the 6 nuScenes cameras -> one image
└── data_construction/             # OPTIONAL: how the verifier dataset was built (no need to run)
```

## Datasets (download from HuggingFace)

| Config dataset name | File | Purpose |
|---|---|---|
| `drive_lmm` | `DriveLMMo1_TRAIN_LLaMA_Factory.json` | Warm-up SFT (DriveLMM-o1 CoT, `<think>/<answer>`) |
| `verifier` | `verifier.json` | Verifier SFT — scores **+ critique** (GT positives + curated hard negatives) |
| `drive_lmm_no_cot` | `DriveLMMo1_TRAIN_LLaMA_Factory_no_cot.json` | Stage-3 Student SFT (answer only) |

## Train

```bash
# 1. Register the datasets: merge configs/dataset_info_snippet.json into
#    LLaMA-Factory's data/dataset_info.json, and set each file_name to your
#    downloaded file.

# 2. Warm-up SFT of the base policy
llamafactory-cli train stage1_verifier/configs/warmup_sft.yaml

# 3. SFT of the verifier
llamafactory-cli train stage1_verifier/configs/verifier_sft.yaml
```

Hyperparameters (both): LoRA rank 64 / alpha 128, vision tower frozen,
lr `1e-4`, 2 epochs, global batch 128 (4 × grad-accum 4 × 8 GPUs) — matching the
paper. The warm-up SFT checkpoint feeds [Stage 2](../stage2_rl); the verifier is
served there via `serve_verifier.sh`.

<details>
<summary><b>How the verifier dataset was built</b> (optional, no need to run)</summary>

The verifier training set is GT positives + hard negatives that are mined from a
baseline policy, scored, meta-verified by Qwen3-VL-235B, and adjudicated. The
full pipeline is in [`data_construction/`](data_construction/) for transparency —
but the curated dataset is released, so you never need to run it.
</details>
