"""Stage 1 - Drop redundant valid positives after meta-verification.

Keeps the useful training signal: valid negatives (Meta says the score was
correct but < 1.0) and disputed/invalid judgments; discards samples the
meta-verifier confirmed as perfect 1.0/1.0/1.0 (redundant positives).

Edit the paths in the CONFIG section below before running.
"""
import json
import os
import re
from tqdm import tqdm

# ================= CONFIG (edit these) =================
# Input: final output of the meta-verifier
INPUT_FILE = '/path/to/data/meta_verifier/meta_verified_results.json'

# Output: filtered data (mostly valid negatives + invalid judgments)
OUTPUT_FILE = '/path/to/data/meta_verifier/meta_verified_results_filter1.json'

def extract_scores(verifier_text):
    """Extract the three scores from a verifier_output string."""
    try:
        p_match = re.search(r'"perception_score":\s*([0-9.]+)', verifier_text)
        l_match = re.search(r'"logic_score":\s*([0-9.]+)', verifier_text)
        s_match = re.search(r'"safety_score":\s*([0-9.]+)', verifier_text)

        p = float(p_match.group(1)) if p_match else None
        l = float(l_match.group(1)) if l_match else None
        s = float(s_match.group(1)) if s_match else None

        return p, l, s
    except:
        return None, None, None

def main():
    print(f"Loading meta-verified data: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    total_count = len(data)
    kept_data = []

    # Counters
    stats = {
        "discarded_valid_1.0": 0,  # discard: Meta says correct AND all 1.0 (redundant)
        "kept_valid_negative": 0,  # keep: Meta says correct but score < 1.0 (gold negative)
        "kept_invalid": 0,         # keep: Meta says incorrect (junior verifier was wrong)
        "kept_parse_error": 0      # keep: parse failure
    }

    print(f"Total samples: {total_count}")
    print("Filtering: discard IF (is_valid=True AND all scores=1.0)...")

    for item in tqdm(data):
        # 1. Meta result
        meta_res = item.get('meta_result', {})
        is_valid = meta_res.get('is_valid')  # True/False/None

        # 2. Junior verifier scores
        verifier_out = item.get('verifier_output', "")
        p, l, s = extract_scores(verifier_out)

        # 3. Decision logic

        # Case A: Meta says valid
        if is_valid is True:
            # Perfect 1.0 sample?
            if p == 1.0 and l == 1.0 and s == 1.0:
                # -> discard (valid positive, redundant)
                stats["discarded_valid_1.0"] += 1
                continue
            else:
                # -> keep (valid negative - 0.5 or 0.0)
                stats["kept_valid_negative"] += 1
                kept_data.append(item)

        # Case B: Meta says invalid
        elif is_valid is False:
            # -> keep (for further inspection; keep to see the distribution)
            stats["kept_invalid"] += 1
            kept_data.append(item)

        # Case C: parse error or other
        else:
            stats["kept_parse_error"] += 1
            kept_data.append(item)

    # Save
    print("-" * 40)
    print(f"Discarded (redundant valid positives): {stats['discarded_valid_1.0']}")
    print(f"Kept (valid negatives - GOLD):         {stats['kept_valid_negative']}")
    print(f"Kept (invalid judgments):              {stats['kept_invalid']}")
    print(f"Kept (parse/meta errors):              {stats['kept_parse_error']}")
    print("-" * 40)
    print(f"Total kept: {len(kept_data)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(kept_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
