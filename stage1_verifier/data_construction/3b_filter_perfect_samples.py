"""Stage 1 - Filter out perfect-score samples from the scored responses.

Samples the verifier scored as perfect (1.0/1.0/1.0) carry no learning signal,
so they are discarded; imperfect/negative samples (and unparsable ones) are
kept for meta-verification.

Edit the paths in the CONFIG section below before running.
"""
import json
import os
import re
from tqdm import tqdm

# ================= CONFIG (edit these) =================
# Input: full data scored by the untuned verifier
INPUT_FILE = '/path/to/data/scored_results/scored_responses.json'

# Output: filtered data (imperfect samples + negatives only)
OUTPUT_FILE = '/path/to/data/scored_results/scored_responses_filtered.json'

def extract_scores(verifier_text):
    """Extract the three scores from a verifier_output string.

    Uses regex rather than json.loads for robustness against extra text the
    LLM may emit around the JSON.
    """
    try:
        # Match "key": value, tolerant of whitespace/newlines.
        p_match = re.search(r'"perception_score":\s*([0-9.]+)', verifier_text)
        l_match = re.search(r'"logic_score":\s*([0-9.]+)', verifier_text)
        s_match = re.search(r'"safety_score":\s*([0-9.]+)', verifier_text)

        # Convert to float when matched, else None.
        p_score = float(p_match.group(1)) if p_match else None
        l_score = float(l_match.group(1)) if l_match else None
        s_score = float(s_match.group(1)) if s_match else None

        return p_score, l_score, s_score
    except Exception as e:
        return None, None, None

def main():
    print(f"📂 Loading data from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    total_count = len(data)
    kept_data = []
    discarded_count = 0
    parse_error_count = 0

    print(f"📊 Total samples before filtering: {total_count}")
    print("🚀 Filtering out perfect scores (1.0/1.0/1.0)...")

    for item in tqdm(data):
        verifier_out = item.get('verifier_output', "")

        # Extract scores.
        p, l, s = extract_scores(verifier_out)

        # Rule:
        # 1. If all three scores parse and are 1.0 -> discard.
        if p == 1.0 and l == 1.0 and s == 1.0:
            discarded_count += 1
            continue

        # 2. Otherwise keep:
        #    - negatives containing 0.5 or 0.0
        #    - unparsable samples (kept so meta-verifier / a human can review)
        else:
            if p is None or l is None or s is None:
                parse_error_count += 1

            kept_data.append(item)

    # Save.
    print("-" * 30)
    print(f"Discarded (perfect 1.0s): {discarded_count}")
    print(f"Parse errors (kept):      {parse_error_count}")
    print(f"Kept (negatives/imperfect): {len(kept_data)}")
    print("-" * 30)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(kept_data, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered data to: {OUTPUT_FILE}")
    print("Next: run the meta-verifier on this file.")

if __name__ == "__main__":
    main()