"""Stage 3 - Latent Thought Distillation (train the Student).

The Student is trained with a joint objective:

    L_total = L_CE + alpha * L_align

where L_CE is standard SFT cross-entropy on prompt->answer pairs (no CoT), and
L_align = 1 - cosine(h_student_answer, h_teacher_think) aligns the Student's
hidden state at the answer anchor with the Teacher's cached final </think> state.

Edit the paths in the CONFIG section below before running.
Launch with DeepSpeed, e.g.:  deepspeed train_distill.py
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from functools import partial

# ================= 1. CONFIG (edit these) =================
MODEL_PATH = "/path/to/models/Qwen3-VL-8B-base"                                  # Student init (base VLM)
TRAIN_JSON = "/path/to/LLaMA-Factory/data/DriveLMMo1_TRAIN_no_cot.json"          # prompt->answer (no CoT)
HIDDEN_PT_PATH = "/path/to/distill/teacher_answer_hidden_states.pt"             # Teacher states (Stage 3 step 1)
OUTPUT_DIR = "/path/to/models/student_lora"                                      # output LoRA dir

# ================= 2. Dataset =================
class HybridDistillDataset(Dataset):
    def __init__(self, json_path, hidden_pt_path, processor):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        print("Loading Teacher feature vectors...")
        raw_teacher_states = torch.load(hidden_pt_path, map_location="cpu")
        self.teacher_states = {k: v.to(torch.bfloat16) for k, v in raw_teacher_states.items()}

        self.processor = processor
        self.processed_data = []
        image_counters = {}

        print("Building dataset index and keeping only samples with a Teacher vector...")
        for item in tqdm(self.raw_data):
            img_path = item["images"][0]
            base_name = os.path.basename(img_path).replace(".png", "")
            count = image_counters.get(base_name, 1)
            data_id = f"{base_name}_{count}"
            image_counters[base_name] = count + 1

            t_v = self.teacher_states.get(data_id)
            if t_v is not None:
                self.processed_data.append({
                    "id": data_id,
                    "image_path": img_path,
                    "question": item["instruction"].replace("<image>\n", "").strip(),
                    "answer": item["output"],
                    "teacher_vector": t_v
                })

        print(f"Done. Kept {len(self.processed_data)} distillation samples.")

    def __len__(self): return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        common_kwargs = {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}

        prompt_msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": item["question"]}]}]
        full_msg = prompt_msg + [{"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_msg, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **common_kwargs
        )
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = self.processor.apply_chat_template(
            full_msg, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", **common_kwargs
        )

        data = {}
        for k, v in inputs.items():
            if k in ["pixel_values", "image_grid_thw"]:
                data[k] = v
            else:
                data[k] = v.squeeze(0)

        labels = data["input_ids"].clone()
        labels[:prompt_len] = -100  # mask the prompt; supervise the answer only
        data["labels"] = labels
        data["teacher_vector"] = item["teacher_vector"]
        return data

# ================= 3. Collate =================
def hybrid_collate_fn(batch, processor):
    input_ids = torch.nn.utils.rnn.pad_sequence([d["input_ids"] for d in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence([d["labels"] for d in batch], batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence([d["attention_mask"] for d in batch], batch_first=True, padding_value=0)
    teacher_vectors = torch.stack([d["teacher_vector"] for d in batch])

    res = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "teacher_vectors": teacher_vectors}
    if "pixel_values" in batch[0]:
        res["pixel_values"] = torch.cat([d["pixel_values"] for d in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        res["image_grid_thw"] = torch.cat([d["image_grid_thw"] for d in batch], dim=0)
    return res

# ================= 4. Custom Trainer =================
class DistillTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_matches = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_vectors = inputs.pop("teacher_vectors", None)
        input_ids = inputs.get("input_ids")

        outputs = model(**inputs, output_hidden_states=True)
        sft_loss = outputs.loss

        if torch.isnan(sft_loss):
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        hidden_states = outputs.hidden_states[-1]
        distill_loss = torch.tensor(0.0).to(model.device).to(torch.float32)
        distill_count = 0

        for b in range(input_ids.shape[0]):
            t_v = teacher_vectors[b]
            if t_v.abs().sum() > 1e-6:
                idx = self._find_custom_anchor(input_ids[b])
                if idx >= 0 and idx < hidden_states.shape[1]:
                    s_v = hidden_states[b, idx, :]

                    # Cast to FP32 for numerical stability.
                    s_v_fp32 = s_v.to(torch.float32)
                    t_v_fp32 = t_v.to(model.device).to(torch.float32)

                    # Alignment loss: 1 - cosine similarity (instead of MSE).
                    cos_sim = torch.nn.functional.cosine_similarity(s_v_fp32, t_v_fp32, dim=-1)
                    batch_cos_loss = 1.0 - cos_sim.mean()

                    if not torch.isnan(batch_cos_loss):
                        distill_loss += batch_cos_loss
                        distill_count += 1

        self.cumulative_matches += distill_count

        # alpha (lambda in the paper) weights the alignment loss.
        alpha = 0.5
        final_distill_loss = (distill_loss / distill_count) if distill_count > 0 else torch.tensor(0.0).to(model.device).to(torch.float32)

        total_loss = sft_loss + alpha * final_distill_loss.to(sft_loss.dtype)

        if self.state.global_step % 1 == 0 and self.args.local_rank <= 0:
            print(f"\n[Step {self.state.global_step}] SFT: {sft_loss.item():.4f} | Distill(Cos): {final_distill_loss.item():.6f} | Match: {distill_count}/{input_ids.shape[0]} | Cumulative Match: {self.cumulative_matches}")

        return (total_loss, outputs) if return_outputs else total_loss

    def _find_custom_anchor(self, input_ids_tensor):
        """Student-side anchor: the assistant header 'assistant\\n' (77091, 198)."""
        ids = input_ids_tensor.tolist()
        for i in range(len(ids) - 1):
            if ids[i] == 77091 and ids[i+1] == 198:
                return i + 1  # anchor at the newline
        return -1

# ================= 5. Main =================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map
    )

    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
        if local_rank <= 0:
            print("Vision tower frozen.")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    if local_rank <= 0:
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = HybridDistillDataset(TRAIN_JSON, HIDDEN_PT_PATH, processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,     # 4 * grad_accum 4 * 8 GPUs = 128 global batch (paper)
        gradient_accumulation_steps=4,
        learning_rate=2e-5,                # LoRA lr (paper)
        num_train_epochs=2,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        max_grad_norm=5.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        deepspeed="ds_config_zero2.json",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = DistillTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=partial(hybrid_collate_fn, processor=processor)
    )

    trainer.train()

if __name__ == "__main__":
    main()
