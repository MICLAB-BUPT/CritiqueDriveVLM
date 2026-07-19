"""Stage 1 - Score the generated responses with the (untuned) verifier.

Each generated agent response is scored across perception / logic / safety by a
verifier model served locally. Multi-GPU via one vLLM worker per GPU.

Edit the paths in the CONFIG section below before running.
"""
import argparse
import json
import os
import math
import time
import re
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from multiprocessing import Process, set_start_method, Manager

# ================= CONFIG (edit these) =================

# 1. Verifier model path (untuned verifier = an Instruct model)
MODEL_PATH = '/path/to/models/Qwen3-VL-8B-Instruct'

# 2. Input data (output of the previous step, containing model_outputs)
INPUT_DATA = '/path/to/data/gen_results_negative/gen_responses.json'

# 3. Output directory (scoring results)
OUTPUT_DIR = '/path/to/data/scored_results'

# 4. Verifier system prompt (scoring rubric)
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
Evaluate the reasoning below and output the JSON object.
"""

# Qwen3-VL resolution config
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# GPUs to use
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

# Batch size (prompts per generate call)
MINI_BATCH_SIZE = 32

# ===========================================

def run_worker(gpu_id, data_chunk, chunk_index, progress_queue):
    """Scoring worker process bound to a single GPU."""
    # 1. GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 2. Init processor and vLLM
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=False,
            mm_processor_kwargs={
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            },
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] Init failed: {e}")
        return

    # 3. Sampling params (Temperature=0 for stable, objective scoring)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,  # enough to emit the JSON
    )

    # 4. Processing loop
    total_items = len(data_chunk)
    chunk_results = []

    # Iterate in steps of MINI_BATCH_SIZE
    for i in range(0, total_items, MINI_BATCH_SIZE):
        batch_raw = data_chunk[i : i + MINI_BATCH_SIZE]

        batch_inputs = []
        batch_metas = []

        for item in batch_raw:
            try:
                # --- A. Parse the instruction to extract the Question ---
                raw_instruction = item.get('instruction', '')
                if "Question:" in raw_instruction:
                    # Split on "Question:" and take the last part.
                    question = raw_instruction.split("Question:")[-1].strip()
                else:
                    # Fallback: strip the <image> tag.
                    question = raw_instruction.replace("<image>", "").strip()

                # --- B. Load image ---
                image_path = item['images'][0]
                image = Image.open(image_path).convert("RGB")

                gt_output = item.get('gt_output', '')

                # --- C. Iterate over the 4 responses generated for this sample ---
                model_outputs = item.get('model_outputs', [])

                for agent_resp in model_outputs:
                    # --- D. Regex-extract reasoning and final answer ---
                    reasoning = ""
                    final_answer = ""

                    think_match = re.search(r"<think>(.*?)</think>", agent_resp, re.DOTALL)
                    answer_match = re.search(r"<answer>(.*?)</answer>", agent_resp, re.DOTALL)

                    if think_match:
                        reasoning = think_match.group(1).strip()
                    if answer_match:
                        final_answer = answer_match.group(1).strip()

                    # Fallback: if the model ignored the format, treat all text as the answer.
                    if not final_answer:
                        final_answer = agent_resp.strip()
                        if not reasoning:
                            reasoning = "No explicit reasoning provided."

                    # --- E. Build the verifier user prompt ---
                    # Format strictly aligned with the fine-tuning data.
                    user_text = f"Question: {question}\n\nAgent's Reasoning:\n{reasoning}\n\nAgent's Final Answer:\n{final_answer}\n\nEvaluate:"

                    # Build chat messages
                    messages = [
                        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},  # vLLM handles the placeholder
                                {"type": "text", "text": user_text}
                            ]
                        }
                    ]

                    # Render prompt string
                    prompt_text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    batch_inputs.append({
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image}
                    })

                    # Record metadata (flattened)
                    batch_metas.append({
                        "images": [image_path],
                        "question": raw_instruction,  # keep the full instruction for later use
                        "gt_output": gt_output,
                        "model_output": agent_resp    # the raw full response
                    })

            except Exception as e:
                print(f"[GPU {gpu_id}] Skip item: {e}")
                continue

        # --- F. Run inference ---
        if batch_inputs:
            try:
                outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)

                for output, meta in zip(outputs, batch_metas):
                    verifier_resp = output.outputs[0].text.strip()

                    # Save result: flattened, one record per response.
                    chunk_results.append({
                        "images": meta["images"],
                        "question": meta["question"],
                        "gt_output": meta["gt_output"],
                        "model_output": meta["model_output"],  # the object being evaluated
                        "verifier_output": verifier_resp        # the verifier's scoring
                    })

            except Exception as e:
                print(f"[GPU {gpu_id}] Inference error: {e}")

        # Update progress (by number of original scenarios processed)
        progress_queue.put(len(batch_raw))

    # 5. Save this GPU's chunk
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, f"scored_chunk_{chunk_index}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, indent=2, ensure_ascii=False)

    print(f"[GPU {gpu_id}] Chunk finished. Saved to {output_path}")


def main():
    # spawn is required for CUDA + multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    print(f"Loading input data: {INPUT_DATA}")
    with open(INPUT_DATA, 'r') as f:
        data_list = json.load(f)

    total_items = len(data_list)
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(total_items / num_gpus)

    print(f"Total scenarios: {total_items}")
    print(f"NOTE: total inferences ~ {total_items * 4} (n=4 per scenario)")

    manager = Manager()
    progress_queue = manager.Queue()

    processes = []
    for i, gpu_id in enumerate(GPU_IDS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        data_chunk = data_list[start_idx:end_idx]

        if not data_chunk:
            continue

        p = Process(target=run_worker, args=(gpu_id, data_chunk, i, progress_queue))
        processes.append(p)
        p.start()

    # Progress bar
    print("Starting multi-GPU scoring pipeline...")
    pbar = tqdm(total=total_items, desc="Scoring Progress", unit="scenario")

    processed_count = 0
    while True:
        alive_count = sum([p.is_alive() for p in processes])

        # Drain the queue
        while not progress_queue.empty():
            try:
                update_val = progress_queue.get_nowait()
                pbar.update(update_val)
                processed_count += update_val
            except:
                break

        if alive_count == 0 and progress_queue.empty():
            break
        time.sleep(0.5)

    pbar.close()
    for p in processes:
        p.join()

    # Merge chunks
    print("Merging results...")
    final_results = []
    for i in range(num_gpus):
        chunk_path = os.path.join(OUTPUT_DIR, f"scored_chunk_{i}.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, 'r') as f:
                final_results.extend(json.load(f))

    final_file = os.path.join(OUTPUT_DIR, "scored_responses.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"All done. Full results saved to: {final_file}")
    print(f"   Total evaluations: {len(final_results)}")

if __name__ == "__main__":
    main()
