"""Stage 3 - Extract the Teacher's final </think> hidden states.

For each training sample we run the frozen Teacher over its full (possibly
multi-turn) reasoning trajectory and cache the last-layer hidden state at the
final </think> token. These vectors are the distillation targets that the
Student is aligned to (see train_distill.py). Multi-GPU via torch.multiprocessing.

Edit the paths in the CONFIG section below before running.
"""
import os
import json
import torch
import math
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch.multiprocessing as mp

# ================= CONFIG (edit these) =================
MODEL_PATH = "/path/to/models/Qwen3-VL-8B-teacher"                       # frozen Teacher
INPUT_JSON_PATH = "/path/to/results/teacher_train_trajectories.json"     # Teacher trajectories
OUTPUT_PT_PATH = "/path/to/distill/teacher_answer_hidden_states.pt"      # output tensor dict
IMAGE_ROOT = "/path/to/data/nuscenes/stitched_output"                    # stitched images

NUM_GPUS = 8
BATCH_SIZE = 2

# Qwen3-VL </think> token id
THINK_END_TOKEN_ID = 151668

def extract_worker(rank, world_size, data_list, output_dir):
    """Worker process bound to a single GPU."""
    # Use the physical rank as the device to avoid all workers piling onto GPU 0.
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Worker started, loading model onto cuda:{rank}...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Place the model on this worker's physical GPU.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map={"": rank},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.eval()

    chunk_data = data_list[rank::world_size]
    print(f"[GPU {rank}] Assigned {len(chunk_data)} records, extracting...")

    hidden_states_dict = {}
    disable_tqdm = (rank != 0)

    with torch.no_grad():
        for i in tqdm(range(0, len(chunk_data), BATCH_SIZE), desc=f"GPU {rank} Progress", disable=disable_tqdm):
            batch_items = chunk_data[i : i + BATCH_SIZE]

            batch_texts = []
            batch_images = []
            valid_ids = []

            for item in batch_items:
                data_id = item["id"]
                base_name = "_".join(data_id.split("_")[:-1])
                image_name = f"{base_name}.png"
                image_path = os.path.join(IMAGE_ROOT, image_name)

                if not os.path.exists(image_path):
                    continue

                image = Image.open(image_path).convert("RGB")

                is_single_turn = (item.get("draft_answer") == item.get("answer")) or (item.get("critique") == "None")
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item["question"]}
                    ]}
                ]

                if is_single_turn:
                    messages.append({"role": "assistant", "content": item["answer"]})
                else:
                    refinement_instruction = (
                        f"### Safety Audit Feedback: {item['critique']}\n\n"
                        "Based on this feedback, please re-examine the provided image and provide a corrected answer. "
                        "Follow a structured and logical reasoning process. "
                        "Organize your response using the format, ensuring each step builds upon the previous one and clearly addresses the hazards mentioned in the feedback. "
                        "Your answer should be structured as Reasoning Steps: (step by step reasoning) Final Answer: (final answer)"
                    )
                    messages.append({"role": "assistant", "content": item["draft_answer"]})
                    messages.append({"role": "user", "content": refinement_instruction})
                    messages.append({"role": "assistant", "content": item["answer"]})

                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

                batch_texts.append(text_prompt)
                batch_images.append(image)
                valid_ids.append(data_id)

            if not batch_texts:
                continue

            # --- Move batch to this GPU (limit pixels to avoid OOM) ---
            inputs = processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            last_layer_hidden_states = outputs.hidden_states[-1]

            for b_idx in range(len(valid_ids)):
                input_ids_list = inputs.input_ids[b_idx].tolist()

                # --- Locate the final </think> token ---
                target_idx = -1
                ids = input_ids_list
                for idx in range(len(ids) - 1, 0, -1):
                    if ids[idx] == THINK_END_TOKEN_ID:
                        target_idx = idx
                        break

                if target_idx != -1:
                    hidden_states_dict[valid_ids[b_idx]] = last_layer_hidden_states[b_idx, target_idx, :].detach().cpu()

                # Validity check (skip NaNs)
                if target_idx != -1 and target_idx < last_layer_hidden_states.shape[1]:
                    feature = last_layer_hidden_states[b_idx, target_idx, :].detach().cpu()
                    if not torch.isnan(feature).any():
                        hidden_states_dict[valid_ids[b_idx]] = feature
                else:
                    if rank == 0:
                        print(f"Warning: sample {valid_ids[b_idx]} has no </think> token")

            # Free the cache each step to reduce VRAM fragmentation.
            torch.cuda.empty_cache()

    part_path = os.path.join(output_dir, f"temp_part_{rank}.pt")
    torch.save(hidden_states_dict, part_path)
    print(f"[GPU {rank}] Done! Extracted {len(hidden_states_dict)} tensors.")

def main():
    print(f"Loading data from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Total records to process: {len(data_list)}")

    output_dir = os.path.dirname(OUTPUT_PT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLaunching {NUM_GPUS}-GPU extraction...")
    mp.spawn(
        extract_worker,
        args=(NUM_GPUS, data_list, output_dir),
        nprocs=NUM_GPUS,
        join=True
    )

    print("\nAll GPUs done, merging results...")
    final_dict = {}
    for rank in range(NUM_GPUS):
        part_path = os.path.join(output_dir, f"temp_part_{rank}.pt")
        if os.path.exists(part_path):
            part_dict = torch.load(part_path)
            final_dict.update(part_dict)
            os.remove(part_path)

    torch.save(final_dict, OUTPUT_PT_PATH)
    print(f"Done. Extracted {len(final_dict)} tensor features in total.")
    print(f"Saved to: {OUTPUT_PT_PATH}")

if __name__ == "__main__":
    main()
