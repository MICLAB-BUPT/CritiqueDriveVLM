<div align="center">

# CritiqueDriveVLM: From Verifier-Guided Reinforcement Learning to Latent Thought Distillation for Autonomous Driving

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Venue](https://img.shields.io/badge/ECCV-2026-1b6fb3.svg)](https://eccv.ecva.net/)

Zhaohong Liu<sup>1</sup>, Hao Ye<sup>1</sup>, Xianlin Zhang<sup>2</sup>, Mengshi Qi<sup>\*1</sup>

<sup>1</sup> State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications, China<br>
<sup>2</sup> School of Digital Media & Design Arts, Beijing University of Posts and Telecommunications, China

</div>

---

## News

- Our paper is accepted to **ECCV 2026**.

---

## Overview

**CritiqueDriveVLM** is a unified three-stage framework that internalizes reasoning directly into a Vision-Language Model (VLM) for autonomous driving. It resolves the reliability-efficiency trade-off of standard SFT and tool-augmented Chain-of-Thought (CoT) approaches:

- **Stage 1 — Warm-up SFT & Verifier Construction.** Enforces a structural reasoning format (`<think>...</think><answer>...</answer>`) and trains an independent multi-dimensional verifier over perception, logic, and safety.
- **Stage 2 — Critique-Driven Multi-Turn RL.** Uses verifier feedback and a step-decay multi-turn penalty under GRPO to cultivate a reliable System-2 Teacher.
- **Stage 3 — Latent Thought Distillation.** Aligns the Student's `<answer>` hidden state with the Teacher's final `</think>` hidden state, internalizing deep reasoning into a fast, CoT-free System-1 Student.

On the **DriveLMM-o1** benchmark, our Teacher boosts Multiple Choice Quality (MCQ) from 55.54% to **76.54%**, while the distilled Student reaches 68.59% MCQ using only **~28 tokens**, reducing inference latency by **88%** (3482 ms → **416 ms**).

---

## Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/MICLAB-BUPT/CritiqueDriveVLM.git
cd CritiqueDriveVLM
uv sync --extra build
```

---

## Acknowledgments

This project builds upon the following open-source works:

- [verl](https://github.com/volcengine/verl) — RL framework for LLMs
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Efficient fine-tuning framework
- [DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1) — Step-by-step reasoning benchmark for driving

---

## Citation

```bibtex
@inproceedings{liu2026critiquedrivevlm,
  title     = {CritiqueDriveVLM: From Verifier-Guided Reinforcement Learning to Latent Thought Distillation for Autonomous Driving},
  author    = {Liu, Zhaohong and Ye, Hao and Zhang, Xianlin and Qi, Mengshi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
