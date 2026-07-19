"""Stage 2 - Composite reward for Critique-Driven Multi-Turn RL (verl).

compute_score() is registered as verl's custom_reward_function. It evaluates the
final turn of a rollout:

    R = W_FORMAT (format compliance)
      + W_ACCURACY * R_acc (MCQ answer correctness)
      + W_VERIFIER * (perception + logic + safety) / 3     [R_verif]
      - (attempts - 1) * DECAY_PER_ATTEMPT                  [multi-turn penalty]

The verifier scores come from the frozen verifier served via vLLM (see
verifier_client.py); its URL can be set with the VERIFIER_API_URL env var.
"""
import re
import asyncio
import logging
import os
from reward.verifier_client import VerifierClient

logger = logging.getLogger(__name__)

# ==============================================================
# 0. Verifier endpoint (override with the VERIFIER_API_URL env var)
# ==============================================================
VERIFIER_API_URL = os.environ.get(
    "VERIFIER_API_URL", "http://localhost:8000/v1/chat/completions"
)

# ==============================================================
# 1. Reward weights
# ==============================================================
W_FORMAT = 0.1      # format score (floor)
W_ACCURACY = 1.0    # final-answer accuracy score
W_VERIFIER = 0.5    # R_verif weight: process verifier (perception/logic/safety)

# Efficiency decay
DECAY_PER_ATTEMPT = 0.2  # each extra attempt decays the core score by 20%

# ==============================================================
# 2. Helper functions
# ==============================================================
def extract_last_assistant_content(solution_str):
    """Extract the last assistant turn from a multi-turn dialogue.

    Slices from the last <think> to the last </answer>. This matters because
    extract_final_answer must not match an earlier (wrong) turn's answer.
    """
    # 1. Start: the last <think> (lower() for robustness against <Think>).
    content_lower = solution_str.lower()
    start_idx = content_lower.rfind("<think>")

    # 2. End: the last </answer>.
    end_idx = content_lower.rfind("</answer>")

    # 3. Slice.
    if start_idx != -1 and end_idx != -1:
        # Ensure the closing tag comes after the opening tag.
        if end_idx > start_idx:
            # +9 to include the </answer> tag itself.
            slice_end = end_idx + 9
            return solution_str[start_idx : slice_end].strip()

    # --- Fallbacks ---
    # If there is a <think> but no </answer> (e.g. truncated generation), still
    # return from <think> onward so the format check below correctly assigns 0.0.
    if start_idx != -1:
        return solution_str[start_idx:].strip()

    # No <think> at all: format is fully broken; return raw text -> scored 0.
    return solution_str.strip()

def extract_final_answer(text):
    # Remove <think>
    text_cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Remove <answer> tags
    text_cleaned = re.sub(r'</?answer>', '', text_cleaned, flags=re.IGNORECASE).strip()

    options = [
        "The final answer is:", "**Final Answer:**", "Final Answer", "Answer",
        "Why take this action?:", "**Final Answer**", "**Final Decision**:",
        "Final Step:", "<CONCLUSION>", "Decision:"
    ]
    final_ans = ""
    found_split = False
    for opt in options:
        if opt in text_cleaned:
            final_ans = text_cleaned.split(opt)[-1]
            found_split = True
            break
    if not found_split: final_ans = text_cleaned
    return final_ans.lstrip(" :*").strip()

def extract_mcq_letter(text):
    if "none of the" in text.lower(): return 'F'
    match = re.search(r'([A-F])[\)\.]', text)
    if match: return match.group(1).upper()
    match = re.search(r'\b([A-F])\b', text)
    if match: return match.group(1).upper()
    return None

# ==============================================================
# 3. Single verifier call
# ==============================================================
async def single_verifier_query(image_path, question, answer, ground_truth):
    """Create a fresh VerifierClient per call (the client manages its own session)."""
    client = VerifierClient(api_url=VERIFIER_API_URL)
    return await client.query(image_path, question, answer, ground_truth)

# ==============================================================
# 4. Main entry point
# ==============================================================
def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    # 1. Extract the last turn for scoring (avoids matching an earlier answer).
    last_response = extract_last_assistant_content(solution_str)

    # 2. Prepare data.
    payload = extra_info if extra_info else {}
    gt_val = ground_truth.get('ground_truth', '') if isinstance(ground_truth, dict) else str(ground_truth)

    image_path = None
    if 'image_path' in payload:
        image_path = payload['image_path']
    elif 'images_payload' in payload and len(payload['images_payload']) > 0:
        image_path = payload['images_payload'][0].get('image')
    elif 'interaction_kwargs' in payload:
        image_path = payload['interaction_kwargs'].get('image_path')

    # Question extraction
    question = None
    if 'question' in payload:
        question = payload['question']
    elif 'question_payload' in payload:
        question = payload['question_payload']
    elif 'interaction_kwargs' in payload:
        question = payload['interaction_kwargs'].get('question')

    # Infer is_mcq from the GT format if not provided.
    is_mcq = payload.get('is_mcq')
    if is_mcq is None:
        is_mcq = bool(re.match(r'^[A-F]$', gt_val.strip(), re.IGNORECASE))

    # --- Step 1: Count attempts by counting feedback messages ---
    # Do not rely on payload['attempts']; count the feedback markers directly.
    cnt_content_err = solution_str.count("### Safety Audit Feedback:")
    cnt_format_err = solution_str.count("[Format Error]")

    # Current attempt = number of past failures + 1
    attempts = cnt_content_err + cnt_format_err + 1

    # --- Step 2: Format check (0.1) ---
    # Format is a hard gate: broken format -> 0.
    format_pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    if not re.search(format_pattern, last_response, re.DOTALL | re.IGNORECASE):
        return 0.0

    # --- Step 3: Content score ---
    base_score = 0.0
    agent_ans = extract_final_answer(last_response)  # from the last turn

    verif_score = 0.0
    if image_path:
        try:
            stats = asyncio.run(single_verifier_query(image_path, question, last_response, gt_val))
            if stats:
                p = float(stats.get('perception_score', 0))
                l = float(stats.get('logic_score', 0))
                s = float(stats.get('safety_score', 0))
                verif_score = (p + l + s) / 3.0  # mean (0.0~1.0)
        except Exception as e:
            logger.error(f"Verifier failed: {e}")
            verif_score = 0  # fallback on network/service error

    base_score += W_VERIFIER * verif_score

    if is_mcq:
        # === MCQ: letter match only (1.0) ===
        pred_letter = extract_mcq_letter(agent_ans)
        gt_letter = extract_mcq_letter(gt_val)

        # Strict match
        if pred_letter and gt_letter and pred_letter == gt_letter:
            base_score += W_ACCURACY
        else:
            base_score += 0.0

    else:
        base_score += verif_score

    # --- Step 4: Efficiency decay ---
    # attempts=1 -> 1.0, attempts=2 -> 0.8, attempts=3 -> 0.6
    decay_amount = (attempts - 1) * DECAY_PER_ATTEMPT

    # Core score
    core_score = base_score - decay_amount

    # --- Step 5: Final sum ---
    # Total = format score (not decayed) + decayed core score
    total_score = W_FORMAT + core_score

    return max(0.01, total_score)  # keep it positive
