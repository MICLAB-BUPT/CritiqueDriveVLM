"""Stage 1 - Final adjudication of disputed samples.

For samples the meta-verifier flagged as invalid (a dispute), a strong arbiter
model fact-checks the critique against the image + GT and issues final binding
scores. Confirmed corrections overwrite the score and become gold negatives.

Edit the paths in the CONFIG section below before running.
"""
import argparse
import json
import os
import re
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= CONFIG (edit these) =================

# 1. Arbiter model (use the strongest available VLM, e.g. Qwen3-VL-235B)
MODEL_PATH = "/path/to/models/Qwen3-VL-235B-A22B-Instruct-FP8"

# 2. Input file (output of the previous meta-verification filter step)
INPUT_FILE = '/path/to/data/meta_verifier/meta_verified_results_filter1.json'

# 3. Output file (final cleaned dataset)
OUTPUT_FILE = '/path/to/data/meta_verifier/meta_verified_results_filter2.json'

# 4. Final arbiter system prompt
FINAL_ARBITER_SYSTEM_PROMPT = """You are the **Supreme Judge** for Autonomous Driving Data Evaluation.

**Context**:
1. An **Agent** generated a response to a driving scenario.
2. A **Meta Auditor** reviewed it and claimed the Agent made a mistake (the "Critique").

**Your Mission**:
You must determine if the **Meta Auditor's Critique** is factually correct based on the **Driving Image** and **Ground Truth (GT)**.
Then, issue the **Final Binding Scores** according to the standards below.

### 📜 The Scoring Standards (The Law You Must Uphold)

The Junior Verifier was instructed to score the Agent based on these **three strictly discrete dimensions**. You must verify if it followed them:

**1. Perception (Visual Grounding)**
* **1.0 (Accurate)**: Agent correctly identifies key objects AND drivable space (e.g., "The left lane is clear"). No hallucinations.
* **0.5 (Incomplete)**: Agent identifies main hazards but misses spatial details or available maneuvers.
* **0.0 (Hallucination/Miss)**: Agent mentions objects NOT present (Hallucination) or misses critical immediate hazards visible in the Image.

**2. Logic (Reasoning Consistency)**
* **1.0 (Sound)**: The reasoning is logical AND **strongly supports the Final Answer**. The cause-and-effect chain is clear.
* **0.5 (Weak)**: Logic is generic, OR there is a slight disconnect between the reasoning and the Final Answer.
* **0.0 (Flawed)**: Contradictory logic, OR the reasoning argues for one action but the Final Answer chooses another.

**3. Safety (Decision Quality) - CRITICAL**
* **1.0 (Optimal)**: The Final Answer is the **best** option (e.g., changing lanes to avoid a static obstacle when safe). Matches GT's intent.
* **0.5 (Conservative)**: The decision avoids collision but is **suboptimal** (e.g., stopping unnecessarily when a safe lane change is possible). **This is NOT a wrong answer, but a passive one.**
* **0.0 (Dangerous)**: The Final Answer violates traffic rules, creates collision risk, or contradicts GT (e.g., GT="Left", Agent="Right").

---

### ⚖️ The Judgment Logic

**Step 1: Fact-Check the Critique**
* Read the **Meta Auditor's Critique**. It claims the Agent hallucinated or was conservative.
* **Look at the Image**: Is the Critique true?
    * *Scenario A*: Critique says "Agent hallucinated a pedestrian". -> **Check Image**. If no pedestrian, Critique is **TRUE**.
    * *Scenario B*: Critique says "Agent missed the truck". -> **Check Image**. If truck is visible, Critique is **TRUE**.
    * *Scenario C*: Critique says "Agent is wrong". -> **Check Image**. If Agent is actually correct and Critique is nitpicking, Critique is **FALSE**.

**Step 2: Assign Final Scores**
* **Case A: If Critique is TRUE (Valid)**:
    * The Agent made the mistake identified by the Auditor. Apply the appropriate penalty (e.g., Perception=0.0 or Safety=0.5).
* **Case B: If Critique is FALSE (Invalid)**:
    * The Auditor was wrong (e.g., Auditor hallucinated a hazard that isn't there).
    * **CRITICAL**: Do NOT automatically give the Agent 1.0.
    * **You must Evaluate the Agent INDEPENDENTLY**: Compare the Agent's answer to the GT/Image yourself.
        * If the Agent is actually correct -> Give **1.0**.
        * If the Agent made a *different* mistake missed by the Auditor -> Give **0.0** or **0.5**.

### Output Format (JSON ONLY)
{
  "adjudication_reason": "I checked the image. The Meta Auditor claims the Agent hallucinated a pedestrian. Looking at the image, there is indeed NO pedestrian. The Meta Auditor is CORRECT. The Agent failed Perception.",
  "meta_critique_is_correct": true,  // True if Meta Auditor's critique is valid. False if Meta Auditor was wrong.
  "final_scores": {
      "perception_score": 0.0,
      "logic_score": 0.5,
      "safety_score": 0.5
  }
}
"""

# Qwen3-VL resolution config
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# ===========================================

def extract_json(text):
    """Robust JSON extraction: handles markdown code fences and noisy text."""
    try:
        # 1. Try ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # 2. Try the outermost { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return None
    except:
        return None

def main():
    print(f"Initializing vLLM with model: {MODEL_PATH}")
    print("   (tensor-parallel across GPUs)")

    # Init vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,  # shard the large model across GPUs
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Sampling params: Temperature=0.0 for deterministic output
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # 1. Load data
    print(f"Loading data from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # 2. Select "disputes": only items the meta-verifier marked as Invalid (False).
    #    Skip None or True.
    target_data = []
    for item in data:
        meta_res = item.get('meta_result', {})
        if meta_res.get('is_valid') is False:
            target_data.append(item)

    print(f"Total disputes to adjudicate: {len(target_data)}")

    if not target_data:
        print("No disputes found. Exiting.")
        return

    BATCH_SIZE = 64  # tune to your VRAM; a 235B model may need a smaller batch
    final_results = []

    # Counters
    stats = {
        "critique_upheld": 0,    # Case A: Meta was right, Agent was wrong
        "critique_rejected": 0,  # Case B: Meta was wrong
        "final_negatives": 0,    # final score 0.0 or 0.5 (what we want most)
        "final_positives": 0     # final score 1.0
    }

    # 3. Batched inference
    for i in tqdm(range(0, len(target_data), BATCH_SIZE), desc="Adjudicating"):
        batch = target_data[i : i + BATCH_SIZE]
        batch_prompts = []
        batch_metas = []

        for item in batch:
            try:
                # The critique from the previous meta-verification round
                prev_reason = item.get('meta_result', {}).get('reason', 'N/A')

                # Build user content
                user_content_text = f"""
**Case Details**:
1. **Scenario**: {item['question']}
2. **Ground Truth**: {item['gt_output']}
3. **Agent's Answer**: {item['model_output']}

**The Accusation (Meta Auditor's Critique)**:
"{prev_reason}"

----------------
**Task**: Look at the Image. Is the Critique CORRECT? If not, assess the Agent yourself. Give the Final Scores.
"""
                image_path = item['images'][0]
                image = Image.open(image_path).convert("RGB")

                # Build messages
                messages = [
                    {"role": "system", "content": FINAL_ARBITER_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_content_text}
                    ]}
                ]

                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                batch_prompts.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image}
                })
                batch_metas.append(item)

            except Exception as e:
                print(f"Error preparing item: {e}")
                continue

        if not batch_prompts:
            continue

        # Run inference
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Parse results
        for output, original_item in zip(outputs, batch_metas):
            resp_text = output.outputs[0].text.strip()

            res_json = extract_json(resp_text)

            if res_json:
                # Extract fields
                is_critique_correct = res_json.get("meta_critique_is_correct")
                final_scores = res_json.get("final_scores", {})
                adjudication_reason = res_json.get("adjudication_reason")

                # Record the arbiter's reasoning and verdict
                original_item['final_source'] = 'arbiter_judgement'
                original_item['arbiter_reason'] = adjudication_reason
                original_item['critique_validity'] = is_critique_correct

                # The arbiter-suggested scores (for reference / override)
                arbiter_score_str = json.dumps({
                    "perception_score": final_scores.get("perception_score", 1.0),
                    "logic_score": final_scores.get("logic_score", 1.0),
                    "safety_score": final_scores.get("safety_score", 1.0)
                })

                # ====================================================
                # Core: decide whether to overwrite based on validity
                # ====================================================

                if is_critique_correct is True:
                    # Case A: the critique was right -> a confirmed negative / correction.
                    # Action: overwrite verifier_output; use directly for training.
                    original_item['verifier_output'] = arbiter_score_str
                    original_item['status'] = "confirmed_correction"
                    stats["critique_upheld"] += 1

                    # Count negatives
                    if final_scores.get("safety_score", 1.0) < 1.0:
                        stats["final_negatives"] += 1

                else:
                    # Case B: the critique was wrong -> complex case.
                    # Action: keep the original verifier_output (junior score);
                    # store the arbiter suggestion for human review.
                    original_item['arbiter_suggestion'] = arbiter_score_str
                    original_item['status'] = "complex_dispute"
                    stats["critique_rejected"] += 1

                final_results.append(original_item)

            else:
                # Parse failure
                original_item['arbiter_error'] = "JSON Parse Failed"
                # Optionally keep or discard

        # Periodic save (avoid losing work on interruption)
        if len(final_results) % 100 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("-" * 40)
    print("Adjudication complete!")
    print(f"Critique upheld (confirmed negative): {stats['critique_upheld']}")
    print(f"Critique rejected (re-evaluated):     {stats['critique_rejected']}")
    print("-" * 20)
    print(f"Final negatives (score < 1.0):        {stats['final_negatives']} (high-value data)")
    print(f"Final positives (score = 1.0):        {stats['final_positives']} (to be discarded)")
    print("-" * 40)
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
