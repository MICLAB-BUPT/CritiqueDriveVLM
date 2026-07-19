# Stage 3 — Latent Thought Distillation (Student)

Compress the System-2 Teacher into a fast, CoT-free System-1 Student by aligning
the Student's `<answer>` hidden state with the Teacher's fully-converged final
`</think>` hidden state.

```
extract_teacher_hidden_states.py   # cache the Teacher's final </think> states -> .pt
train_distill.py                   # train the Student: L_CE + alpha * L_align
ds_config_zero2.json               # DeepSpeed ZeRO-2 config
```

## Objective

```
L_total = L_CE + alpha * L_align
L_align = 1 - cosine( h_student(answer anchor),  h_teacher(final </think>) )
```

The Student is initialized from the base VLM and trained on **prompt→answer**
pairs with **no CoT text**, so it must rely on its latent state to preserve the
Teacher's reasoning depth.

## Run

```bash
# 1. Cache the Teacher's final </think> hidden states (multi-GPU)
python extract_teacher_hidden_states.py

# 2. Train the Student (DeepSpeed ZeRO-2)
deepspeed train_distill.py
```

The alignment weight is set to the paper value **λ = 0.5** (`alpha` in
`train_distill.py`), and distillation uses LoRA at lr `2e-5` for 2 epochs.
