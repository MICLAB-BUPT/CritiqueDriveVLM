# Inference

Run the trained models on DriveLMM-o1.

```
infer_teacher.py    # multi-turn, verifier-guided Teacher (Stage 2)
infer_student.py    # CoT-free, low-latency Student (Stage 3)
infer_latency.py    # latency / token profiling of the Teacher pipeline
```

Each script has a `CONFIG` section / CLI args (`--model_path`, `--gpus`, ...).
`infer_teacher.py` and `infer_latency.py` require the frozen verifier to be
running (see [../stage2_rl/serve_verifier.sh](../stage2_rl/serve_verifier.sh));
the endpoint is read from `VERIFIER_API_URL` (default `http://localhost:8000`).

```bash
# Teacher (multi-turn, verifier-guided)
python infer_teacher.py  --model_path /path/to/Qwen3-VL-8B-teacher  --gpus 0,1,2,3

# Student (CoT-free, fast)
python infer_student.py  --model_path /path/to/Qwen3-VL-8B-student  --gpus 0,1,2,3

# Latency benchmark (single GPU)
python infer_latency.py  --model_path /path/to/Qwen3-VL-8B-teacher  --gpus 0
```

Outputs are JSON files (gitignored) consumed by [../evaluation](../evaluation).
