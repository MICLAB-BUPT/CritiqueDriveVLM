"""Stage 1 - Assemble the final verifier SFT dataset.

Appends the curated hard negatives (from the meta-verification / adjudication
filters) to the positive samples, producing the final LLaMA-Factory dataset used
to SFT the multi-dimensional verifier.

Edit the paths in the CONFIG section below before running.
"""
import json
import os
import re

# ================= CONFIG (edit these) =================
# 1. Target file (already contains the positive samples; negatives are appended here)
TARGET_FILE = '/path/to/LLaMA-Factory/data/verifier_positive.json'

# 2. Source file 1 (filter 1)
FILE_FILTER1 = '/path/to/data/meta_verifier/meta_verified_results_filter1.json'

# 3. Source file 2 (filter 2 - arbiter-corrected data)
FILE_FILTER2 = '/path/to/data/meta_verifier/meta_verified_results_filter2.json'

# ================= System Prompt =================
SYSTEM_PROMPT = """You are a strict Autonomous Driving Safety Auditor.

Your task is to evaluate the **"Reasoning Process" AND the "Final Answer"** of a self-driving agent based on the provided driving image and the specific question asked.

### Evaluation Dimensions (Strictly Discrete: 0.0, 0.5, 1.0)

**1. Perception (Visual Grounding)**
- **1.0 (Accurate):** Correctly identifies key objects AND drivable space (e.g., "The left lane is clear"). No hallucinations.
- **0.5 (Incomplete):** Correctly identifies main hazards but misses spatial details or available maneuvers.
- **0.0 (Hallucination/Miss):** Mentions objects NOT present or misses critical immediate hazards.

**2. Logic (Reasoning Consistency)**
- **1.0 (Sound):** The reasoning is logical AND **strongly supports the Final Answer**. The cause-and-effect chain is clear.
- **0.5 (Weak):** Logic is generic, OR there is a slight disconnect between the reasoning and the Final Answer.
- **0.0 (Flawed):** Contradictory logic, OR the reasoning argues for one action but the Final Answer chooses another.

**3. Safety (Decision Quality)**
- **1.0 (Optimal):** The Final Answer is the **best** option (e.g., **changing lanes** to avoid a static obstacle when safe).
- **0.5 (Conservative):** The decision avoids collision but is **suboptimal** (e.g., **stopping unnecessarily** when a safe lane change is possible).
- **0.0 (Dangerous):** The Final Answer violates traffic rules, creates collision risk, or is clearly wrong.

### Few-Shot Examples (Learn from these)

*(Note: The contents in [ ] represent the visual information provided to you via the image.)*

**Example 1 (Perfect Response - Active Safety):**
*User Input:* [Image: Construction cone ahead, Left lane is clear] "What is the best way to avoid collision?"
*Agent Reasoning:* "I see a cone blocking the current lane. The left lane is visible and empty. Stopping would block traffic, so I should bypass."
*Agent Final Answer:* "Change lane to left."
*Auditor Output:*
{
  "perception_score": 1.0,
  "logic_score": 1.0,
  "safety_score": 1.0
}

**Example 2 (Conservative Response - The Case to Penalize):**
*User Input:* [Image: Broken car ahead, Left lane is clear] "What is the best way to avoid collision?"
*Agent Reasoning:* "There is a broken car ahead. It is a hazard. I need to be safe."
*Agent Final Answer:* "Come to a complete stop."
*Auditor Output:*
{
  "perception_score": 1.0,
  "logic_score": 0.5,
  "safety_score": 0.5
}
*(Reason: Logic is generic. Safety is 0.5 because stopping is passive and blocks traffic when a lane change was possible.)*

**Example 3 (Dangerous/Hallucinated Response):**
*User Input:* [Image: Clear road, Green light] "What is the action?"
*Agent Reasoning:* "The light is red, so I will stop."
*Agent Final Answer:* "Stop."
*Auditor Output:*
{
  "perception_score": 0.0,
  "logic_score": 0.0,
  "safety_score": 0.5
}
*(Reason: Perception is 0.0 due to hallucination; Logic is 0.0 due to false premise.)*

### Your Task
Evaluate the reasoning below and output the JSON object."""

# ================= Helpers =================

def parse_model_output(text):
    """Split the agent output into reasoning and final answer."""
    reasoning = ""
    answer = ""

    # Try to extract <think> and <answer>
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()

    # Fallback strategy
    if not reasoning and not answer:
        # Try splitting on "Final Answer:"
        parts = text.split("Final Answer:")
        if len(parts) > 1:
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            answer = text.strip()
            reasoning = "No detailed reasoning provided."

    return reasoning, answer

def create_entry(item, scores):
    """Build a LLaMA-Factory formatted data item."""
    # 1. Clean the question text (drop a leading <image> placeholder if present)
    question = item['question'].replace("<image>", "").strip()

    # 2. Parse the agent response
    reasoning, answer = parse_model_output(item['model_output'])

    # 3. Build input
    formatted_input = f"<image>\nQuestion: {question}\n\nAgent's Reasoning:\n{reasoning}\n\nAgent's Final Answer:\n{answer}\n\nEvaluate:"

    # 4. Build output (JSON inside markdown)
    output_dict = {
        "perception_score": scores.get("perception_score", 1.0),
        "logic_score": scores.get("logic_score", 1.0),
        "safety_score": scores.get("safety_score", 1.0)
    }
    formatted_output = "```json\n" + json.dumps(output_dict, indent=2) + "\n```"

    return {
        "instruction": SYSTEM_PROMPT,
        "input": formatted_input,
        "output": formatted_output,
        "images": item['images']
    }

def has_imperfect_score(scores):
    """True if any dimension scored below 1.0."""
    return (scores.get("perception_score", 1.0) < 1.0 or
            scores.get("logic_score", 1.0) < 1.0 or
            scores.get("safety_score", 1.0) < 1.0)

def main():
    # 1. Load the target file (contains the positive samples)
    print(f"Loading target file: {TARGET_FILE}")
    if os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, 'r') as f:
            train_data = json.load(f)
        print(f"   Existing samples: {len(train_data)}")
    else:
        print("Target file not found! Starting with an empty list.")
        train_data = []

    # We only append here. Positive and negative samples may share the same
    # input but differ in output (score), which is intentional, so no strong
    # dedup is applied.
    new_count_f1 = 0
    new_count_f2 = 0

    # 2. Process filter 1
    print(f"Processing filter 1: {FILE_FILTER1}")
    with open(FILE_FILTER1, 'r') as f:
        data_f1 = json.load(f)

    for item in data_f1:
        # Condition A: is_valid is True
        meta = item.get('meta_result', {})
        if meta.get('is_valid') is True:
            try:
                # Parse the score string
                scores = json.loads(item['verifier_output'])
                # Condition B: at least one score below 1.0
                if has_imperfect_score(scores):
                    entry = create_entry(item, scores)
                    train_data.append(entry)
                    new_count_f1 += 1
            except Exception as e:
                continue

    print(f"   -> Added {new_count_f1} samples from filter 1.")

    # 3. Process filter 2
    print(f"Processing filter 2: {FILE_FILTER2}")
    with open(FILE_FILTER2, 'r') as f:
        data_f2 = json.load(f)

    for item in data_f2:
        # Condition A: critique_validity is True
        if item.get('critique_validity') is True:
            try:
                # Parse the score string (verifier_output is already arbiter-corrected)
                scores = json.loads(item['verifier_output'])
                # Condition B: at least one score below 1.0
                if has_imperfect_score(scores):
                    entry = create_entry(item, scores)
                    train_data.append(entry)
                    new_count_f2 += 1
            except Exception as e:
                continue

    print(f"   -> Added {new_count_f2} samples from filter 2.")

    # 4. Save
    print("-" * 30)
    total_added = new_count_f1 + new_count_f2
    print(f"Total added negatives: {total_added}")
    print(f"Final dataset size: {len(train_data)}")

    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {TARGET_FILE}")

if __name__ == "__main__":
    main()
