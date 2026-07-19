import os
import argparse
import json
import time
import requests
import base64
import math
import re
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import random

"""Inference with the multi-turn, verifier-guided Teacher (Stage 2) via vLLM.

Round 1: the Agent produces a draft. The frozen verifier scores it and, if the
answer is wrong (MCQ) or imperfect (open-ended), a critique triggers Round 2 in
which the Agent refines its answer. Requires the verifier server running (see
../stage2_rl/serve_verifier.sh).

Paths default to placeholders; override via CLI args or edit ds_collections.
"""

# ================= CONFIG =================
# 1. Agent model / dataset config (vLLM)
ds_collections = {
    'DriveLMMo1': {
        'root': '/path/to/data/DriveLMMo1_TRAIN.json',
        'image_root': '/path/to/data/nuscenes/stitched_output',
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    }
}

# 2. Verifier service (must be running in the background; VERIFIER_API_URL env var respected)
VERIFIER_API_URL = os.environ.get("VERIFIER_API_URL", "http://localhost:8000/v1/chat/completions")
VERIFIER_MODEL_NAME = "verifier"  # served-model-name of the verifier

def clean_response(text):
    """Strip <think>/<answer> tags."""
    if not isinstance(text, str): return str(text)
    return re.sub(r'</?(think|answer)>', '', text, flags=re.IGNORECASE).strip()

def extract_final_answer(text):
    """Return the content after a 'Final Answer' marker."""
    options = ["The final answer is:","**Final Answer:**","Final Answer:", "Final Answer",
               "Answer:", "**Final Decision**:","Final Step:", "<CONCLUSION>"]
    text = clean_response(text)
    for opt in options:
        if opt in text:
            return text.split(opt)[-1].strip()
    return text

def extract_option_letter(text):
    """Extract the MCQ option letter (A-E).

    Handles formats like: "A)", "(A)", "Option A", "The answer is A".
    """
    if not text: return None
    # Prefer explicit "A)" / "B)" formats.
    pattern = r'([A-E])\)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].upper()

    # Otherwise, a lone letter near the end (e.g. "Final Answer: A").
    match = re.search(r'\b([A-E])\b', text[-10:])
    if match:
        return match.group(1).upper()

    return None

# ================= Helpers: image processing & API call =================

def smart_resize_for_api(image_path):
    """Load an image and base64-encode it (JPEG) for the verifier API."""
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1003520
    
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        num_pixels = width * height
        
        target_width = width
        target_height = height
        
        if num_pixels > MAX_PIXELS:
            ratio = math.sqrt(MAX_PIXELS / num_pixels)
            target_width = int(width * ratio)
            target_height = int(height * ratio)
        elif num_pixels < MIN_PIXELS:
            ratio = math.sqrt(MIN_PIXELS / num_pixels)
            target_width = int(width * ratio)
            target_height = int(height * ratio)
            
        if target_width != width or target_height != height:
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_verifier(question, agent_answer, gt_answer, image_path):
    """Call the verifier to get scores + critique."""
    # System prompt for the verifier (coach mode).
    SYSTEM_PROMPT = """
You are a strict Autonomous Driving Safety Coach.

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
    
    user_content = f"Question: {question}\n\nAgent Answer: {agent_answer}"
    
    try:
        base64_img = smart_resize_for_api(image_path)
        payload = {
            "model": VERIFIER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }
        
        # Timeout for stability
        response = requests.post(VERIFIER_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']

            # Clean and parse JSON
            clean_content = content.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(clean_content)
                # Fill missing score fields with 0
                return {
                    "perception_score": float(data.get("perception_score", 0.0)),
                    "logic_score": float(data.get("logic_score", 0.0)),
                    "safety_score": float(data.get("safety_score", 0.0)),
                    "critique": data.get("critique", "None")
                }
            except:
                return {"perception_score": 0.0, "logic_score": 0.0, "safety_score": 0.0, "critique": "None"}
        else:
            return {"perception_score": 0.0, "logic_score": 0.0, "safety_score": 0.0, "critique": "None"}
            
    except Exception as e:
        print(f"Verifier Call Failed: {e}")
        return {"perception_score": 0.0, "logic_score": 0.0, "safety_score": 0.0, "critique": "None"}

# ================= Main =================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path/to/models/Qwen3-VL-8B-teacher')
    parser.add_argument('--datasets', type=str, default='DriveLMMo1')
    parser.add_argument('--outdir', type=str, default='results_reflexion')
    parser.add_argument('--sample_size', type=int, default=2000)  # number of samples
    parser.add_argument('--gpus', type=str, default="0,1,2,3")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=50, help='chunk size (keep small; two rounds per chunk)')
    return parser.parse_args()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tp_size = len(args.gpus.split(','))

    # 1. Init processor
    print(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. Init vLLM (Agent)
    ds_config = ds_collections.get(args.datasets)
    print(f"Initializing Agent vLLM with TP={tp_size}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": ds_config['min_pixels'],
            "max_pixels": ds_config['max_pixels'],
        },
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # 3. Load data
    print(f"Loading dataset...")
    with open(ds_config['root'], 'r', encoding='utf-8') as f:
        full_data = json.load(f)  # training set is standard JSON

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    output_path = os.path.join(args.outdir, f"Reflexion_Qwen3VL_train_{time_prefix}.json")
    
    base_instruction = "When answering the question based on the provided image, follow a structured and logical reasoning process. Organize your response using the format, ensuring each step builds upon the previous one and clearly explains how the image(s) contribute to the solution. Your answer should be structured as Reasoning Steps: (step by step reasoning) Final Answer: (final answer) \n Question: "
    
    final_results = []
    chunk_size = args.chunk_size
    total_items = len(full_data)
    
    print(f"Starting reflexion pipeline (chunk size: {chunk_size})")

    with tqdm(total=total_items, desc="Processing") as pbar:
        for i in range(0, total_items, chunk_size):
            batch_raw = full_data[i : i + chunk_size]

            # ================= Round 1: Agent draft =================
            batch_inputs_r1 = []
            batch_metas = []

            for item in batch_raw:
                try:
                    # Training set uses 'idx' instead of 'id'.
                    data_id = item.get('idx', 'unknown')

                    # Image path: the stitched image is named by the first ID segments.
                    # e.g. idx "73030..._c36e..._1" -> image "73030..._c36e....png"
                    parts = data_id.split('_')
                    base_name = "_".join(parts[:-1]) if len(parts) > 1 else data_id
                    image_filename = f"{base_name}.png"

                    question_text = item.get('question', '')
                    gt_answer = item.get('answer', '')
                    gt_answer = extract_final_answer(gt_answer)

                    full_question = f"{base_instruction} {question_text}"
                    image_path = os.path.join(ds_config['image_root'], image_filename)
                    image = Image.open(image_path).convert("RGB")

                    # Build vLLM input
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": full_question}
                        ]}
                    ]
                    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    batch_inputs_r1.append({
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image}
                    })
                    batch_metas.append({
                        "id": data_id,
                        "question": full_question,
                        "question_pure": question_text,  # for the verifier
                        "gt_answers": gt_answer,
                        "image_path": image_path,
                        "image_obj": image,  # cached for Round 2
                        "messages_history": messages  # cached dialogue history
                    })
                except Exception as e:
                    print(f"Error prep item {item.get('id')}: {e}")

            if not batch_inputs_r1:
                pbar.update(len(batch_raw))
                continue

            # Run Round 1 Inference
            outputs_r1 = llm.generate(batch_inputs_r1, sampling_params)
            
           # ================= Verifier step & Round 2 prep =================
            batch_inputs_r2 = []
            indices_needing_r2 = []

            for idx, (output, meta) in enumerate(zip(outputs_r1, batch_metas)):
                draft_answer = output.outputs[0].text
                meta["draft_answer"] = draft_answer

                # --- 1. Query the verifier for a detailed report ---
                report = call_verifier(
                    question=meta["question_pure"],
                    agent_answer=draft_answer,
                    gt_answer=meta["gt_answers"],
                    image_path=meta["image_path"]
                )
                
                # Save scores and critique
                meta["perception_score"] = report["perception_score"]
                meta["logic_score"] = report["logic_score"]
                meta["safety_score"] = report["safety_score"]
                meta["critique"] = report["critique"]

                # --- 2. Gating (mirror the Stage-2 RL loop): go to Round 2 when the
                #        verifier is NOT perfect across all dimensions. ---
                is_perfect_score = (meta["perception_score"] >= 1.0 and
                                    meta["logic_score"] >= 1.0 and
                                    meta["safety_score"] >= 1.0)

                should_refine = not is_perfect_score
                if should_refine:
                    # Exactly the critique feedback the Teacher saw during RL
                    # (see stage2_rl/interaction/reflexion_interaction.py).
                    refinement_instruction = (
                        f"### Safety Audit Feedback: {meta['critique']}\n"
                        "Based on this feedback, please re-examine the provided image and provide a corrected answer. "
                        "Follow a structured and logical reasoning process. "
                        "Organize your response using the format, ensuring each step builds upon the previous one and clearly addresses the hazards mentioned in the feedback. "
                        "Your answer should be structured as <think> (step by step reasoning) </think> <answer> (final answer) </answer>\n\n"
                    )

                # --- 3. Refine or finalize ---
                if should_refine:

                    new_messages = meta["messages_history"] + [
                        {"role": "assistant", "content": draft_answer},
                        {"role": "user", "content": refinement_instruction}
                    ]
                    
                    prompt_text_r2 = processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)
                    
                    batch_inputs_r2.append({
                        "prompt": prompt_text_r2,
                        "multi_modal_data": {"image": meta["image_obj"]}
                    })
                    indices_needing_r2.append(idx)
                else:
                    # No refinement needed (MCQ correct, open-ended perfect, or no critique).
                    meta["final_answer"] = draft_answer

            # ================= Round 2: Agent refinement (if any) =================
            if batch_inputs_r2:
                outputs_r2 = llm.generate(batch_inputs_r2, sampling_params)

                # Write refined answers back.
                for i, r2_out in enumerate(outputs_r2):
                    original_idx = indices_needing_r2[i]
                    meta = batch_metas[original_idx]
                    meta["final_answer"] = r2_out.outputs[0].text  # refined answer

           # ================= Save results =================
            for meta in batch_metas:
                final_results.append({
                    "id": meta["id"],
                    "question": meta["question"],
                    "gt_answers": meta["gt_answers"],
                    "scores": {
                        "perception": meta.get("perception_score", 0.0),
                        "logic": meta.get("logic_score", 0.0),
                        "safety": meta.get("safety_score", 0.0)
                    },
                    "draft_answer": meta.get("draft_answer"),
                    "critique": meta.get("critique"),
                    "answer": meta.get("final_answer")
                })
            
            pbar.update(len(batch_raw))

            # Periodic save
            if i > 0 and i % (chunk_size * 5) == 0:
                 with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"Reflexion done. Saved to {output_path}")

if __name__ == "__main__":
    main()