"""Stage 1 - Meta-verification of the junior verifier's judgments.

A much stronger model (Qwen3-VL-235B) audits whether the junior verifier scored
each agent response correctly, producing an is_valid verdict + reasoning. This
is the automated part of the "filtering & human verification" curation step.

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

# 1. Meta-verifier model (a strong, large VLM)
MODEL_PATH = "/path/to/models/Qwen3-VL-235B-A22B-Instruct-FP8"

# 2. Input data (junior-verifier scored responses)
INPUT_FILE = '/path/to/data/scored_results/scored_responses.json'

# 3. Output file
OUTPUT_FILE = '/path/to/data/meta_verifier/meta_verified_results.json'

# 4. Meta-verifier prompt (reason first -> verdict last)
META_VERIFIER_SYSTEM_PROMPT = """You are the Supreme Quality Assurance Auditor for an Autonomous Driving AI system.

**Your Mission**:
You are auditing the judgment of a "Junior AI Verifier".
You must determine if the Junior Verifier correctly applied the **Scoring Standards** to the **Agent's Answer** based on the **Ground Truth (GT)** and the **Driving Image**.

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

### ⚖️ Your Judgment Logic

**Input Data**:
1. **Image**: The visual truth.
2. **Question**: The task.
3. **Ground Truth**: The correct answer (Standard).
4. **Agent's Answer**: The candidate response.
5. **Junior Verifier's Judgment**: The Score and Critique given by the Junior model.

**How to Decide `is_valid`**:

* **VALID (True)**:
    * The Junior Verifier gave **1.0** for a perfect match in all dimensions.
    * The Junior Verifier gave **0.5** because the Agent was indeed Conservative (Safety) or had weak reasoning (Logic).
    * The Junior Verifier gave **0.0** because the Agent hallucinated (Perception) or was dangerous (Safety).

* **INVALID (False)**:
    * **False Leniency**: Agent was passive (Stop) when GT was active (Go), but Junior Verifier gave Safety **1.0** (Should be 0.5).
    * **False Penalty**: Agent was correct, but Junior Verifier gave it **0.0** or **0.5** without valid reason.
    * **Hallucination Miss**: Agent mentioned objects not in the Image, but Junior Verifier gave Perception **1.0**.
    * **Logic Failure**: Agent's reasoning contradicted its answer, but Junior Verifier gave Logic **1.0**.

### Output Format (JSON ONLY)
Please analyze the case first, then give your verdict.

{
  "reason": "Step-by-step analysis comparing Agent, GT, and Image. Explain why the Verifier's score is correct or incorrect based on the Standards.",
  "is_valid": true  // Set true if Verifier is correct, false if incorrect.
}
"""

# Qwen3-VL resolution config
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# ===========================================

def main():
    print("Initializing vLLM with the meta-verifier model (tensor-parallel)...")

    # Init vLLM. tensor_parallel_size shards the large model across GPUs.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
        mm_processor_kwargs={
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
    )

    # Init processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Sampling params: Temperature=0; large max_tokens to leave room for "reason".
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    print(f"Loading scored data: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data_list = json.load(f)

    # Verify everything here; optionally restrict to non-1.0 samples.
    target_data = data_list

    print(f"Total samples to verify: {len(target_data)}")

    # Batched inference
    BATCH_SIZE = 64

    final_results = []

    for i in tqdm(range(0, len(target_data), BATCH_SIZE), desc="Meta Verify Progress"):
        batch = target_data[i : i + BATCH_SIZE]

        batch_prompts = []
        batch_metas = []

        for item in batch:
            try:
                # Build the input text
                user_content_text = f"""
Please audit this evaluation:

**1. Scenario Question**:
{item['question']}

**2. Ground Truth (Standard)**:
{item['gt_output']}

**3. Agent's Answer (Candidate)**:
{item['model_output']}

**4. Junior Verifier's Judgment (Score & Critique)**:
{item['verifier_output']}
"""
                image_path = item['images'][0]
                image = Image.open(image_path).convert("RGB")

                # Build chat messages
                messages = [
                    {"role": "system", "content": META_VERIFIER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_content_text}
                        ]
                    }
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
                print(f"Error processing item: {e}")
                continue

        if not batch_prompts:
            continue

        # Run inference
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Parse results
        for output, original_item in zip(outputs, batch_metas):
            meta_response = output.outputs[0].text.strip()

            # Try to extract JSON
            try:
                json_match = re.search(r"\{.*\}", meta_response, re.DOTALL)
                if json_match:
                    meta_json = json.loads(json_match.group(0))
                else:
                    # No JSON found -> mark as parse error
                    meta_json = {"is_valid": False, "reason": "Parse Error: " + meta_response}
            except:
                meta_json = {"is_valid": False, "reason": "JSON Parse Error"}

            # Attach the meta result to the original record
            original_item['meta_result'] = meta_json
            final_results.append(original_item)

        # Periodic save (every 100 records)
        if len(final_results) % 100 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Meta verification complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
