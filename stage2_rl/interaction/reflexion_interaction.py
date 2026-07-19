"""Stage 2 - Multi-turn critique interaction loop (verl BaseInteraction).

At each turn the frozen verifier scores the policy's response and, if it is not
good enough (and the max-attempt limit K is not reached), returns a natural
language critique that is appended to the dialogue so the policy can refine its
answer. A step-decay penalty discourages relying on extra turns.
"""
import logging
import re
import asyncio
from uuid import uuid4
from typing import Dict, Any, List, Tuple, Optional
from verl.interactions.base import BaseInteraction
from reward.verifier_client import VerifierClient

logger = logging.getLogger(__name__)

# ==========================================================
# Helper functions (kept consistent with the reward function)
# ==========================================================
def extract_final_answer(text):
    text_cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
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

def check_format_strict(text):
    """Strict format check: must contain both <think> and <answer> tags."""
    format_pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    return bool(re.search(format_pattern, text, re.DOTALL | re.IGNORECASE))

# ==========================================================
# Interaction class
# ==========================================================
class ReflexionInteraction(BaseInteraction):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instance_dict = {}
        self.verifier = VerifierClient()  # uses VERIFIER_API_URL env var / localhost:8000
        self.max_attempts = config.get("max_attempts", 2)
        self.penalty_weight = 0.2  # each extra attempt costs 0.2
        print(f"[ReflexionInteraction] Initialized! Max Attempts: {self.max_attempts}")

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # --- Robust field extraction ---
        ik = kwargs.get("interaction_kwargs", {})
        image_path = kwargs.get("image_path") or ik.get("image_path")
        question = kwargs.get("question") or ik.get("question") or ""
        ground_truth = kwargs.get("ground_truth") or ik.get("ground_truth") or ""
        is_mcq = kwargs.get("is_mcq")
        if is_mcq is None:
            is_mcq = ik.get("is_mcq") or any(x in str(question) for x in ["A)", "B)", "Choose from"])

        self._instance_dict[instance_id] = {
            "attempts": 0,
            "ground_truth": ground_truth,
            "image_path": image_path,
            "question": question,
            "is_mcq": is_mcq,
            "last_score": 0.0,
        }
        return instance_id

    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, Dict[str, Any]]:
        instance = self._instance_dict[instance_id]
        instance["attempts"] += 1
        curr_attempt = instance["attempts"]

        # 1. Extract the latest assistant response
        content = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                content = item.get("content", "")
                break

        # 2. Format check
        if not check_format_strict(content):
            print(f"[Format Error] Instance {instance_id[:8]} | No proper tags found.")
            if curr_attempt >= self.max_attempts:
                return True, "Done", 0.0, {}
            else:
                return (curr_attempt >= self.max_attempts), "**[Format Error]** Your response must strictly follow this structure:\n<think>\n(Step-by-Step Reasoning)\n</think>\n<answer>\n(Final Answer)\n</answer>", 0.0, {}
        
        # 3. Init scores (guards against UnboundLocalError below)
        p, l, s = 0.0, 0.0, 0.0
        v_score = 0.0
        critique = ""

        # 4. Query the verifier
        stats = await self.verifier.query(
            instance["image_path"], instance["question"], content, instance["ground_truth"]
        )
        
        if stats:
            p = float(stats.get('perception_score', 0.0))
            l = float(stats.get('logic_score', 0.0))
            s = float(stats.get('safety_score', 0.0))
            v_score = (p + l + s) / 3.0
            critique = stats.get('critique', "")

        # 5. Content score = R_acc + alpha * (s_per + s_log + s_saf)   (paper Eq. 2)
        is_mcq = instance["is_mcq"]

        # alpha * verifier term (alpha folded into the 0.5 weight)
        r_content = 0.5 * v_score

        if is_mcq:
            # R_acc for MCQ: strict final-answer letter match against the ground truth.
            agent_ans = extract_final_answer(content)
            pred_letter = extract_mcq_letter(agent_ans)
            gt_letter = extract_mcq_letter(str(instance["ground_truth"]))
            mcq_correct = bool(pred_letter and gt_letter and pred_letter == gt_letter)
            r_content += 1.0 if mcq_correct else 0.0
        else:
            # R_acc for open-ended: no letter to match, so use the verifier score as
            # the final-answer alignment proxy.
            r_content += v_score

        # 6. Final step reward = R_fmt + r_content - P_mt (step-decay penalty)
        penalty = (curr_attempt - 1) * self.penalty_weight
        final_step_reward = max(0.01, 0.1 + r_content - penalty)
        instance["last_score"] = final_step_reward

        # 7. Termination (paper Sec. 3.2): verifier perfect across ALL dimensions, or max turns.
        is_perfect = (p >= 1.0 and l >= 1.0 and s >= 1.0)
        should_terminate = is_perfect or curr_attempt >= self.max_attempts

        # 7. Return
        if should_terminate:
            meta = {"attempts": curr_attempt, "is_mcq": is_mcq}
            return True, "Done", final_step_reward, meta
        else:
            feedback_msg = (
                f"### Safety Audit Feedback: {critique}\n"
                "Based on this feedback, please re-examine the provided image and provide a corrected answer. "
                "Follow a structured and logical reasoning process. "
                "Organize your response using the format, ensuring each step builds upon the previous one and clearly addresses the hazards mentioned in the feedback. "
                "Your answer should be structured as <think> (step by step reasoning) </think> <answer> (final answer) </answer>\n\n"
            )
            return False, feedback_msg, final_step_reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict.get(instance_id, {}).get("last_score", 0.01)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]