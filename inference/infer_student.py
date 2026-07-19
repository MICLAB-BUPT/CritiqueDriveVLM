"""Inference with the CoT-free Student (Stage 3) via vLLM.

The Student answers directly (no explicit reasoning), so this is the low-latency
System-1 path. Multi-GPU via tensor parallelism inferred from --gpus.

Paths default to placeholders; override via CLI args or edit ds_collections.
"""
import os
import argparse
import json
import time
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= Dataset config =================
ds_collections = {
    'DriveLMMo1': {
        'root': '/path/to/data/DriveLMMo1_TEST.jsonl',
        'image_root': '/path/to/data/nuscenes/stitched_output',
        # Qwen3-VL recommended resolution range
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path/to/models/Qwen3-VL-8B-student', help='model path')
    parser.add_argument('--datasets', type=str, default='DriveLMMo1')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--gpus', type=str, default="0,1,2,3", help='GPU id list, e.g. "0,1,2,3"')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=50, help='chunk size to limit VRAM fragmentation')
    return parser.parse_args()

def main():
    args = get_args()

    # Set visible GPUs and infer tensor-parallel size.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tp_size = len(args.gpus.split(','))  # e.g. "0,1,2,3" -> 4
    print(f"Set CUDA_VISIBLE_DEVICES={args.gpus}, auto-detected TP_SIZE={tp_size}")

    # 1. Init processor (only to render the chat template string)
    print(f"Loading processor from {args.model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: AutoProcessor load failed: {e}")
        return

    # Dataset config
    ds_config = ds_collections.get(args.datasets)
    if not ds_config:
        raise ValueError(f"Dataset {args.datasets} not found in ds_collections")

    # 2. Init vLLM engine
    print(f"Initializing vLLM engine with TP={tp_size}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=8192,          # adjust to your VRAM
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,         # set True if you hit CUDA Graph errors
        # Pass Qwen-VL resolution params through to vLLM.
        mm_processor_kwargs={
            "min_pixels": ds_config['min_pixels'],
            "max_pixels": ds_config['max_pixels'],
        },
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # 3. Load dataset
    print(f"Loading dataset from {ds_config['root']}...")
    with open(ds_config['root'], 'r') as f:
        # Accept both JSONL and a JSON list.
        try:
            data_list = [json.loads(line) for line in f.readlines()]
        except:
            f.seek(0)
            data_list = json.load(f)

    # Output path
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    output_path = os.path.join(args.outdir, f"Qwen3VL_Student_{time_prefix}.json")

    # The Student is distilled on bare (image + question) -> answer pairs with NO
    # system prompt (see stage3_distill). Inference MUST use the same bare prompt
    # so the Student behaves as trained; adding an extra system prompt here would
    # be a train/inference mismatch.

    final_results = []
    chunk_size = args.chunk_size
    total_items = len(data_list)

    print(f"Starting sequential pipeline (total: {total_items}, chunk size: {chunk_size})")
    print("Logic: load a chunk of images -> run GPU -> save -> repeat")

    # 4. Single-threaded chunked loop
    with tqdm(total=total_items, desc="Inference Progress", unit="sample") as pbar:

        for i in range(0, total_items, chunk_size):
            # --- Stage A: CPU prepares a chunk of data ---
            batch_raw = data_list[i : i + chunk_size]
            batch_inputs = []
            batch_metas = []

            for item in batch_raw:
                try:
                    data_id = item.get('id', 'unknown')
                    # Parse question and GT
                    if 'conversations' in item:
                        question_text = item['conversations'][0]['value'].replace("<image>\n", "").strip()
                        gt_answer = item['conversations'][1]['value'].strip()
                    else:
                        question_text = item.get('question', '')
                        gt_answer = item.get('answer', '')

                    full_question = question_text   # bare question, matching Stage-3 training

                    # Load image
                    image_path = os.path.join(ds_config['image_root'], item['image'])
                    image = Image.open(image_path).convert("RGB")

                    # Render prompt string
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": full_question},
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
                        "id": data_id,
                        "question": full_question,
                        "gt_answers": gt_answer
                    })
                except Exception as e:
                    print(f"Error loading item {item.get('id')}: {e}")
                    continue

            # Skip if the whole chunk failed.
            if not batch_inputs:
                pbar.update(len(batch_raw))
                continue

            # --- Stage B: GPU inference (vLLM) ---
            batch_outputs = llm.generate(batch_inputs, sampling_params)

            # --- Stage C: parse & collect ---
            for output, meta in zip(batch_outputs, batch_metas):
                generated_text = output.outputs[0].text
                final_results.append({
                    "id": meta["id"],
                    "question": meta["question"],
                    "answer": generated_text,
                    "gt_answers": meta["gt_answers"]
                })

            pbar.update(len(batch_raw))

            # Periodic save (every 10 chunks) to avoid losing work.
            if i > 0 and i % (chunk_size * 10) == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"Done. All results saved to {output_path}")

if __name__ == "__main__":
    main()
