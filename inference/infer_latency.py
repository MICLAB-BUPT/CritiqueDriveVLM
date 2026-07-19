"""Latency / token profiling for the multi-turn Teacher pipeline.

Times the full Round-1 (+ optional Round-2 refinement) pipeline per sample on a
single GPU and reports average latency, generation length, and TPS. Requires the
verifier server running (see ../stage2_rl/serve_verifier.sh).

Paths default to placeholders; override via CLI args or edit ds_collections.
"""
import os
import argparse
import json
import time
import requests
import base64
import math
import re
import random
import torch
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= CONFIG =================
# 1. Dataset config
ds_collections = {
    'DriveLMMo1': {
        'root': '/path/to/data/DriveLMMo1_TEST.jsonl',
        'image_root': '/path/to/data/nuscenes/stitched_output',
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    }
}

# 2. Verifier service (VERIFIER_API_URL env var respected)
VERIFIER_API_URL = os.environ.get("VERIFIER_API_URL", "http://localhost:8000/v1/chat/completions")
VERIFIER_MODEL_NAME = "verifier"

# ================= Parsing helpers =================

def clean_response(text):
    if not isinstance(text, str): return str(text)
    return re.sub(r'</?(think|answer)>', '', text, flags=re.IGNORECASE).strip()

def extract_final_answer(text):
    options = ["The final answer is:","**Final Answer:**","Final Answer:", "Final Answer", 
               "Answer:", "**Final Decision**:","Final Step:", "<CONCLUSION>"]
    text = clean_response(text)
    for opt in options:
        if opt in text:
            return text.split(opt)[-1].strip()
    return text

def extract_option_letter(text):
    if not text: return None
    pattern = r'([A-E])\)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].upper()
    match = re.search(r'\b([A-E])\b', text[-10:])
    if match:
        return match.group(1).upper()
    return None

def smart_resize_for_api(image_path):
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1003520
    with Image.open(image_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
        width, height = img.size
        num_pixels = width * height
        target_width, target_height = width, height
        if num_pixels > MAX_PIXELS:
            ratio = math.sqrt(MAX_PIXELS / num_pixels)
            target_width, target_height = int(width * ratio), int(height * ratio)
        elif num_pixels < MIN_PIXELS:
            ratio = math.sqrt(MIN_PIXELS / num_pixels)
            target_width, target_height = int(width * ratio), int(height * ratio)
        if target_width != width or target_height != height:
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ================= Verifier call =================

def call_verifier(question, agent_answer, gt_answer, image_path):
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
                {"role": "user", "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            "temperature": 0.0, "max_tokens": 512
        }
        response = requests.post(VERIFIER_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            data = json.loads(response.json()['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip())
            return {
                "perception_score": float(data.get("perception_score", 0.0)),
                "logic_score": float(data.get("logic_score", 0.0)),
                "safety_score": float(data.get("safety_score", 0.0)),
                "critique": data.get("critique", "None")
            }
    except Exception as e:
        return {"perception_score": 0.0, "logic_score": 0.0, "safety_score": 0.0, "critique": "None"}

# ================= Main (latency benchmark) =================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path/to/models/Qwen3-VL-8B-teacher')
    parser.add_argument('--datasets', type=str, default='DriveLMMo1')
    parser.add_argument('--outdir', type=str, default='results_reflexion')
    parser.add_argument('--gpus', type=str, default="3", help='use a single GPU for latency measurement')
    parser.add_argument('--sample_num', type=int, default=100, help='number of random samples to time')
    return parser.parse_args()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tp_size = len(args.gpus.split(','))
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    ds_config = ds_collections.get(args.datasets)
    
    llm = LLM(
        model=args.model_path, tensor_parallel_size=tp_size, trust_remote_code=True,
        max_model_len=8192, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": ds_config['min_pixels'], "max_pixels": ds_config['max_pixels']}
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    with open(ds_config['root'], 'r') as f:
        data_list = [json.loads(line) for line in f.readlines()]
    
    # Randomly sample for timing.
    if len(data_list) > args.sample_num:
        data_list = random.sample(data_list, args.sample_num)

    total_inference_time = 0.0
    total_tokens = 0
    final_results = []
    base_instruction = "When answering the question based on the provided image, follow a structured and logical reasoning process. Your answer should be structured as Reasoning Steps: (step by step reasoning) Final Answer: (final answer) \n Question: "

    print(f"Starting benchmark on GPU {args.gpus} for {len(data_list)} samples...")

    for item in tqdm(data_list, desc="Measuring Inference Time"):
        # --- Pipeline timing start ---
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # 1. Round 1: draft
        image_path = os.path.join(ds_config['image_root'], item['image'])
        image = Image.open(image_path).convert("RGB")
        question_text = item['conversations'][0]['value'].replace("<image>\n", "").strip()
        full_q = f"{base_instruction} {question_text}"
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": full_q}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        out_r1 = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], sampling_params, use_tqdm=False)[0]
        draft = out_r1.outputs[0].text
        total_tokens += len(out_r1.outputs[0].token_ids)

        # 2. Verifier step
        report = call_verifier(question_text, draft, item['conversations'][1]['value'], image_path)

        # 3. Gating (mirror the Stage-2 RL loop / infer_teacher): Round 2 when the
        #    verifier is NOT perfect across all dimensions.
        should_refine = not (report["perception_score"] >= 1.0 and
                             report["logic_score"] >= 1.0 and
                             report["safety_score"] >= 1.0)

        # 4. Round 2: refinement (same critique feedback as training / infer_teacher)
        final_ans = draft
        if should_refine:
            refinement_instruction = (
                f"### Safety Audit Feedback: {report['critique']}\n"
                "Based on this feedback, please re-examine the provided image and provide a corrected answer. "
                "Follow a structured and logical reasoning process. "
                "Organize your response using the format, ensuring each step builds upon the previous one and clearly addresses the hazards mentioned in the feedback. "
                "Your answer should be structured as <think> (step by step reasoning) </think> <answer> (final answer) </answer>\n\n"
            )
            refine_msg = messages + [{"role": "assistant", "content": draft}, {"role": "user", "content": refinement_instruction}]
            prompt_r2 = processor.apply_chat_template(refine_msg, tokenize=False, add_generation_prompt=True)
            out_r2 = llm.generate([{"prompt": prompt_r2, "multi_modal_data": {"image": image}}], sampling_params, use_tqdm=False)[0]
            final_ans = out_r2.outputs[0].text
            total_tokens += len(out_r2.outputs[0].token_ids)

        # --- Pipeline timing end ---
        torch.cuda.synchronize()
        total_inference_time += (time.perf_counter() - start_time)

        final_results.append({"id": item.get('id'), "answer": final_ans})

    # Print the benchmark report.
    avg_time = (total_inference_time / args.sample_num) * 1000
    print(f"\nPerformance report (single GPU):\nAvg inference time: {avg_time:.2f} ms\nAvg generation length: {total_tokens/args.sample_num:.2f} tokens\nTPS: {total_tokens/total_inference_time:.2f} tokens/s")

if __name__ == "__main__":
    main()