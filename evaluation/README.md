# Evaluation

Score model outputs with the DriveLMM-o1 protocol.

```
evaluate.py      # GPT-4o judge scoring (12 reasoning metrics + final answer / MCQ)
token_count.py   # average generation length (tokens) for the efficiency analysis
```

`evaluate.py` uses **GPT-4o** as the judge (the official DriveLMM-o1 protocol).
Set your key and paths, then run:

```bash
export OPENAI_API_KEY=sk-...          # required
# export OPENAI_BASE_URL=...          # optional, for an OpenAI-compatible endpoint

# edit input_file / dataset / output_file in the CONFIG section, then:
python evaluate.py       # -> per-sample scores + a summary report
python token_count.py    # -> avg / median / p95 generated tokens
```

Reported metrics follow the paper: Risk Assessment, Traffic Rule Adherence,
Scene Awareness, Relevance, Missing Details, overall Reasoning score, and MCQ.
