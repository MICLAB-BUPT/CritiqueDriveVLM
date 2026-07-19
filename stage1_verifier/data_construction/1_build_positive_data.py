"""Stage 1 - Build POSITIVE verifier training data from ground-truth CoT.

Ground-truth reasoning/answers are treated as positive samples and assigned
the maximum score across all dimensions (perception/logic/safety = 1.0).
Output is in LLaMA-Factory format, ready to SFT the multi-dimensional verifier.

Edit the paths in the CONFIG section below before running.
"""
import json
import re
import os

# ================= CONFIG (edit these) =================

# 1. Source SFT training data (DriveLMM-o1 CoT annotations)
INPUT_FILE = "/path/to/data/DriveLMMo1_TRAIN_SFT.json"

# 2. Output verifier SFT data (LLaMA-Factory format)
OUTPUT_FILE = "/path/to/LLaMA-Factory/data/verifier_positive.json"

# 3. Verifier system prompt (scoring rubric)
VERIFIER_SYSTEM_PROMPT = """You are a strict Autonomous Driving Safety Auditor.

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

# ================= Processing logic =================

def extract_content(text):
    """Split raw output into <think> (reasoning) and <answer> (final answer)."""
    # Try to extract <think>
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    # Try to extract <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    # Fallback: no tags -> split on keywords (compat with legacy format)
    if not think and "**Step-by-Step Reasoning**" in text:
        parts = text.split("**Final Answer**")
        think = parts[0].replace("**Step-by-Step Reasoning**:", "").strip()
        if len(parts) > 1:
            answer = "**Final Answer**" + parts[1]
            
    return think, answer

def main():
    print(f"Reading source data: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    verifier_dataset = []

    for item in data:
        # 1. Extract the question (drop the system-prompt part, keep Question only).
        #    Assumes the original instruction looks like "<image>\n... Question: xxx".
        raw_instruction = item.get('instruction', '')
        if "Question:" in raw_instruction:
            question = raw_instruction.split("Question:")[-1].strip()
        else:
            # No "Question:" marker -> fall back to the last line.
            question = raw_instruction.split("\n")[-1]

        # 2. Extract the agent (i.e. GT) reasoning and answer.
        original_output = item.get('output', '')
        reasoning, final_answer = extract_content(original_output)

        # Skip samples whose format could not be parsed.
        if not reasoning or not final_answer:
            continue

        # 3. Build the verifier user input (the "input" column).
        #    No <image> tag needed here; LLaMA-Factory handles the images field.
        verifier_input = f"<image>\nQuestion: {question}\n\nAgent's Reasoning:\n{reasoning}\n\nAgent's Final Answer:\n{final_answer}\n\nEvaluate:"

        # 4. Build the verifier output (the "output" column).
        #    Source is GT, so this is a positive sample -> full marks.
        verifier_output = """```json
{
  "perception_score": 1.0,
  "logic_score": 1.0,
  "safety_score": 1.0
}
```"""

        # 5. Assemble a LLaMA-Factory formatted sample.
        new_entry = {
            "instruction": VERIFIER_SYSTEM_PROMPT,  # -> prompt column
            "input": verifier_input,                # -> query column
            "output": verifier_output,              # -> response column
            "images": item.get('images', [])        # -> images column
        }

        verifier_dataset.append(new_entry)

    # Save.
    print(f"Done. Generated {len(verifier_dataset)} positive samples.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(verifier_dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()