"""Count generated answer tokens (for the average-token / latency analysis).

Edit the paths in the CONFIG section below before running.
"""
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= CONFIG (edit these) =================
# Point at a model whose tokenizer matches the generations.
model_path = "/path/to/models/Qwen3-VL-8B"
data_file = "/path/to/results/model_outputs.json"

def count_generated_tokens():
    print(f"Loading tokenizer from {model_path} (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lengths = []
    print(f"Counting generation length over {len(data)} records...")

    for item in tqdm(data):
        answer_text = item.get('answer', "")
        # Count only the answer tokens.
        # add_special_tokens=False so [CLS]/[SEP]/etc. are not counted.
        tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        lengths.append(len(tokens))

    # ================= Stats =================
    lengths = np.array(lengths)
    print("\n" + "=" * 30)
    print("Results (unit: tokens)")
    print(f"Count:      {len(lengths)}")
    print(f"Mean:       {np.mean(lengths):.2f}")
    print(f"Max:        {np.max(lengths)}")
    print(f"Min:        {np.min(lengths)}")
    print(f"Median:     {np.median(lengths)}")
    print(f"95th pct:   {np.percentile(lengths, 95)}")
    print(f"Total:      {np.sum(lengths)}")
    print("=" * 30)

if __name__ == "__main__":
    count_generated_tokens()
