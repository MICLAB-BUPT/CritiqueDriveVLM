"""Stage 1 - Generate HARD NEGATIVES for verifier training.

A baseline policy (GRPO trained only with format/accuracy rewards) runs
inference on the training set with multiple samples per prompt, yielding
erroneous responses (hallucinations / logical contradictions) that serve as
hard negatives for the verifier. Multi-GPU via one vLLM worker per GPU.

Edit the paths in the CONFIG section below before running.
"""
import argparse
import json
import os
import math
import time
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from multiprocessing import Process, set_start_method, Queue, Manager

# ================= CONFIG (edit these) =================
# Baseline policy model (format/accuracy-only GRPO checkpoint)
MODEL_PATH = '/path/to/models/Qwen3-VL-8B-baseline-grpo'
# Input data (SFT training data)
INPUT_DATA = '/path/to/data/DriveLMMo1_TRAIN_SFT.json'
# Output directory
OUTPUT_DIR = '/path/to/data/gen_results_negative'

# Qwen3-VL resolution config
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# GPUs to use
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

# Inference batch size inside each worker. Controls how often the progress bar
# updates; 32-50 is a good trade-off between responsiveness and throughput.
MINI_BATCH_SIZE = 50

def run_worker(gpu_id, data_chunk, chunk_index, progress_queue):
    """Worker process bound to a single GPU."""
    # 1. GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 2. Init processor
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        # On failure, just return so the main process does not deadlock.
        print(f"[GPU {gpu_id}] Processor load failed: {e}")
        return

    # 3. Init vLLM
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
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
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] vLLM Init failed: {e}")
        return

    # 4. Sampling params (multiple samples per prompt to mine diverse negatives)
    sampling_params = SamplingParams(
        n=4,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    # 5. Mini-batch loop (batched so the progress bar can update).
    total_chunk_size = len(data_chunk)
    all_results = []  # all results for this GPU

    for i in range(0, total_chunk_size, MINI_BATCH_SIZE):
        batch_raw = data_chunk[i : i + MINI_BATCH_SIZE]

        # --- Prepare prompts ---
        batch_inputs = []
        batch_metas = []

        for item in batch_raw:
            try:
                raw_instruction = item.get('instruction', '')

                # Keep the full instruction as the user content.
                if "Question:" in raw_instruction:
                    full_user_content = raw_instruction
                else:
                    full_user_content = raw_instruction

                # Image path
                if isinstance(item['images'], list):
                    image_path = item['images'][0]
                else:
                    image_path = item['images']

                image = Image.open(image_path).convert("RGB")

                # Build chat messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": full_user_content},
                        ],
                    }
                ]
                
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                batch_inputs.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image}
                })
                
                batch_metas.append({
                    "instruction": raw_instruction,
                    "images": [image_path],
                    "gt_output": item.get('output', ''),
                })

            except Exception as e:
                print(f"⚠️ [GPU {gpu_id}] Skip item: {e}")
                continue

        # --- Run inference ---
        if batch_inputs:
            try:
                # use_tqdm=False to avoid vLLM's own progress bars cluttering the log.
                outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)

                # --- Collect results ---
                for output, meta in zip(outputs, batch_metas):
                    generated_texts = [o.text.strip() for o in output.outputs]
                    all_results.append({
                        "instruction": meta["instruction"],
                        "images": meta["images"],
                        "gt_output": meta["gt_output"],
                        "model_outputs": generated_texts
                    })
            except Exception as e:
                print(f"❌ [GPU {gpu_id}] Inference Error: {e}")

        # --- Report progress: this batch of len(batch_raw) items is done ---
        progress_queue.put(len(batch_raw))

    # 6. Save this GPU's chunk
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, f"result_chunk_{chunk_index}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"[GPU {gpu_id}] Finished! Saved {len(all_results)} items.")

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    print(f"📂 Loading dataset: {INPUT_DATA}")
    with open(INPUT_DATA, 'r') as f:
        data_list = json.load(f)

    total_items = len(data_list)
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(total_items / num_gpus)

    print(f"📊 Total Data: {total_items} | GPUs: {num_gpus} | Per GPU: ~{chunk_size}")

    # Cross-process queue for progress updates.
    manager = Manager()
    progress_queue = manager.Queue()

    processes = []
    for i, gpu_id in enumerate(GPU_IDS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        data_chunk = data_list[start_idx:end_idx]

        if not data_chunk:
            continue

        # Pass the progress_queue to each worker.
        p = Process(target=run_worker, args=(gpu_id, data_chunk, i, progress_queue))
        processes.append(p)
        p.start()

    # The main process owns the overall progress bar.
    print("Starting multi-GPU inference...")

    pbar = tqdm(total=total_items, desc="Total Progress", unit="sample")

    processed_count = 0
    finished_processes = 0

    while True:
        # Are any child processes still alive?
        alive_count = sum([p.is_alive() for p in processes])

        # Drain progress updates from the queue.
        while not progress_queue.empty():
            try:
                update_val = progress_queue.get_nowait()
                if isinstance(update_val, int):
                    pbar.update(update_val)
                    processed_count += update_val
            except:
                break

        # Exit once all workers are done and the queue is drained.
        if alive_count == 0 and progress_queue.empty():
            break

        time.sleep(0.1)  # avoid busy-spinning the CPU

    pbar.close()

    # Reap all processes.
    for p in processes:
        p.join()

    print("All GPUs finished.")

    # Merge chunks.
    print("Merging results...")
    final_results = []
    for i in range(num_gpus):
        chunk_path = os.path.join(OUTPUT_DIR, f"result_chunk_{i}.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, 'r') as f:
                final_results.extend(json.load(f))
    
    final_file = os.path.join(OUTPUT_DIR, "gen_responses_test.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged {len(final_results)} items to {final_file}")

if __name__ == "__main__":
    main()