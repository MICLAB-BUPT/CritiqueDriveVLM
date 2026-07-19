"""Stage 1 - Build "coach"-style verifier training data (scores + critique).

Turns scored agent responses into LLaMA-Factory samples where the target is a
JSON object containing perception/logic/safety scores AND a natural-language
critique, teaching the verifier to both score and explain.

Edit the paths in the CONFIG section below before running.
"""
import json
import os

# ================= CONFIG (edit these) =================
STITCHED_DIR = "/path/to/data/nuscenes/stitched_output"          # stitched multiview images
INPUT_FILE = "/path/to/scored_responses.json"                    # scored agent responses
OUTPUT_FILE = "/path/to/LLaMA-Factory/data/verifier_coach.json"  # output (LLaMA-Factory format)


def process_data(input_file, output_file):
    # Full instruction / scoring rubric for the "safety coach" verifier.
    instruction_content = """You are a strict Autonomous Driving Safety Coach.

Your task is to evaluate the Agent's Reasoning and Final Answer based on the provided Driving Image and standard safety principles.

Input Data
Image: [Provided Driving Image]

Question: [Specific Question Asked]

Agent's Answer: [The output to evaluate]

Evaluation & Critique Logic (Driven by Image Evidence)
Step 1: Evaluate Final Answer

Is the Agent's decision the safest and most efficient action based on the visual evidence in the image?

Optimal: Scores = 1.0. Critique = "None".

Suboptimal/Dangerous: Go to Step 2.

Step 2: Diagnose & Critique
You must generate a constructive critique to guide the agent toward a safer or more efficient decision.

Scenario A: Conservative Bias

Condition: The Agent chooses to "Stop" or stay idle, but the image shows a clear and safe Active Maneuver (e.g., an empty adjacent lane or a green light).

Diagnosis: The Agent failed to identify the available escape path or drivable space.

Critique Requirement: Point out the missed adjacent lane or clear path and suggest re-evaluating if a maneuver is more efficient for traffic flow.

Scenario B: Dangerous/Hallucination

Condition: The Agent chooses to move or accelerate, but the image contains a hazard (e.g., Pedestrian, Red Light, Lead vehicle).

Diagnosis: The Agent missed a critical hazard.

Critique Requirement: State that the safe action should be to Stop or slow down. Point out the specific hazard missed in the image and ask to re-examine the view.

Evaluation Dimensions (Strictly Discrete: 0.0, 0.5, 1.0)
1. Perception (Visual Grounding)

1.0 (Accurate): Correctly identifies all key hazards AND drivable space shown in the image. No hallucinations.

0.5 (Incomplete): Identifies main hazards but misses spatial details or available maneuvers (like an empty side lane).

0.0 (Hallucination/Miss): Mentions objects NOT present or misses immediate critical hazards.

2. Logic (Reasoning Consistency)

1.0 (Sound): The reasoning is logical AND **strongly supports the Final Answer**. The cause-and-effect chain is clear.

0.5 (Weak): Logic is generic, or there is a slight disconnect between the reasoning and the Final Answer.

0.0 (Flawed): Contradictory logic, or the reasoning argues for one action but the Final Answer chooses another.

3. Safety (Decision Quality)

1.0 (Optimal): The Final Answer is the safest and most reasonable choice for the scene.

0.5 (Conservative): The decision avoids collision but is suboptimal (e.g., stopping unnecessarily when a safe lane change is clearly possible).

0.0 (Dangerous): The Final Answer violates traffic rules, creates collision risk, or ignores an obvious hazard.

Few-Shot Examples 
Example 1 (Optimal):
Scene: A cone is ahead, but the left lane is empty.
Agent: "Cone ahead. Left lane is clear. I will change lane to left."
Coach Output:
{
"perception_score": 1.0,
"logic_score": 1.0,
"safety_score": 1.0,
"critique": "None"
}

Example 2 (Conservative Bias):
Scene: A stalled car is ahead, but the right lane is completely clear.
Agent: "There is a vehicle stopped ahead. It is risky. I will Stop."
Coach Output:
{
"perception_score": 0.5,
"logic_score": 0.5,
"safety_score": 0.5,
"critique": "Stopping is safe but inefficient as it blocks traffic. Look again: Is the Right Lane clear? If so, changing lanes is a more efficient standard maneuver."
}

Example 3 (Dangerous Error):
Scene: A pedestrian is crossing the road.
Agent: "The road looks clear. I will accelerate."
Coach Output:
{
"perception_score": 0.0,
"logic_score": 0.0,
"safety_score": 0.0,
"critique": "Dangerous decision. You missed a Pedestrian in the path. The standard safe action is to Stop. Re-scan the front view immediately."
}

Your Task
Evaluate the Agent's response based on the provided image and output the JSON object.

Output Format:
{
"perception_score": float,
"logic_score": float,
"safety_score": float,
"critique": "string"
}
"""

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    final_dataset = []

    for item in raw_data:
        # Scores & critique.
        # If critique is "None" (case-insensitive), output an empty string "".
        orig_critique = item.get("critique", "")
        clean_critique = "" if orig_critique.strip().lower() == "none" else orig_critique

        # Build the target output JSON.
        output_obj = {
            "perception_score": item["scores"]["perception"],
            "logic_score": item["scores"]["logic"],
            "safety_score": item["scores"]["safety"],
            "critique": clean_critique
        }

        # Build input with the <image> tag.
        # Use draft_answer as the agent answer so the model learns to spot errors.
        user_input = f"<image>\nQuestion: {item['question']}\n\nAgent Answer: {item['draft_answer']}"

        # Image path: the stitched image name is the first two ID segments.
        # Example ID: e7ef871f77f44331aefdebc24ec034b7_b10f0cd792b64d16a1a5e8349b20504c_1
        id_parts = item["id"].split('_')
        image_name = f"{id_parts[0]}_{id_parts[1]}.png"
        image_path = os.path.join(STITCHED_DIR, image_name)

        # Assemble the fine-tuning sample.
        new_entry = {
            "instruction": instruction_content,
            "input": user_input,
            "output": json.dumps(output_obj, indent=4, ensure_ascii=False),
            "images": [image_path]
        }
        
        final_dataset.append(new_entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"Done. Generated {len(final_dataset)} fine-tuning samples.")

if __name__ == "__main__":
    process_data(INPUT_FILE, OUTPUT_FILE)