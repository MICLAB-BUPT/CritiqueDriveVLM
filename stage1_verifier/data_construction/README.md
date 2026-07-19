# Verifier Data Construction (optional — provenance only)

> **You do NOT need to run any of this.** The curated verifier dataset is
> released on HuggingFace — just download it and go straight to
> [`../configs/verifier_sft.yaml`](../configs/verifier_sft.yaml).
>
> These scripts are provided **only to document how the verifier training set
> was built** from the DriveLMM-o1 data. Reproducing them requires a strong
> judge model (Qwen3-VL-235B) and multi-GPU vLLM, and is not necessary to use
> the framework.

## Pipeline (how the released verifier dataset was produced)

```
1_build_positive_data.py      # GT reasoning/answers -> positive samples (all scores = 1.0)
1b_build_coach_data.py        # coach-style samples: scores + natural-language critique

2_gen_hard_negatives.py       # baseline policy generates multiple responses -> hard negatives
3_score_negatives.py          # verifier scores each response (perception/logic/safety)
3b_filter_perfect_samples.py  # drop perfect-1.0 responses (no learning signal)

4_meta_verify.py              # Qwen3-VL-235B audits the junior verifier's scores
4c_filter_valid_positives.py  # drop redundant valid positives; keep gold negatives + disputes
4b_final_adjudication.py      # arbiter fact-checks disputed critiques -> final scores

5_build_verifier_dataset.py   # merge positives + curated negatives -> final verifier SFT set
```

The output is the LLaMA-Factory–format verifier dataset used by
[`../configs/verifier_sft.yaml`](../configs/verifier_sft.yaml).
