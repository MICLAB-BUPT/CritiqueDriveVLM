"""Async client for the frozen multi-dimensional verifier served via vLLM.

The verifier (an OpenAI-compatible chat endpoint) scores an agent response over
perception / logic / safety and returns a critique. Used by both the Stage-2
reward function and the multi-turn interaction loop.

The endpoint defaults to http://localhost:8000/v1/chat/completions and can be
overridden with the VERIFIER_API_URL environment variable.
"""
import os
import aiohttp
import asyncio
import json
import re
import math
import base64
from io import BytesIO
from PIL import Image

DEFAULT_VERIFIER_API_URL = os.environ.get(
    "VERIFIER_API_URL", "http://localhost:8000/v1/chat/completions"
)

class VerifierClient:
    # Optional shared connection pool (see get_session).
    _shared_session = None
    _semaphore = None

    def __init__(self, api_url=DEFAULT_VERIFIER_API_URL, model_name="verifier"):
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = """
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

    def process_image(self, image_path, max_pixels=1003520):
        # Load, downscale if needed, and base64-encode the image as JPEG.
        if not image_path: return None
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                width, height = img.size
                total_pixels = width * height
                if total_pixels > max_pixels:
                    scale = math.sqrt(max_pixels / total_pixels)
                    img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except:
            return None

    async def get_session(self):
        """Get a process-wide shared connection pool (optional helper)."""
        if VerifierClient._shared_session:
            try:
                if VerifierClient._shared_session.closed or VerifierClient._shared_session.loop.is_closed():
                    VerifierClient._shared_session = None
            except Exception:
                # Accessing .loop failed -> session state is broken, reset it.
                VerifierClient._shared_session = None
        # Create a new session if none is available.
        if VerifierClient._shared_session is None:
            # Re-init the semaphore so it binds to the current event loop.
            VerifierClient._semaphore = asyncio.Semaphore(16)

            connector = aiohttp.TCPConnector(limit=500, ttl_dns_cache=300)
            VerifierClient._shared_session = aiohttp.ClientSession(connector=connector)

        return VerifierClient._shared_session

    async def query(self, image_path, question, agent_answer, gt_answer):
        """Async query with rate limiting and retries."""
        # 1. Preprocess the image (outside the semaphore, not a concurrency slot).
        base64_img = self.process_image(image_path)
        if not base64_img: return None

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Question: {question}\n\nAgent Answer: {agent_answer}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            "temperature": 0.1, 
            "max_tokens": 512
        }

        # Timeout applies to the whole request.
        timeout = aiohttp.ClientTimeout(total=120)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use a fresh ClientSession per request to avoid event-loop confusion.
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            full_content = data['choices'][0]['message']['content']
                            
                            try: return json.loads(full_content)
                            except:
                                match = re.search(r"(\{.*\})", full_content, re.DOTALL)
                                if match:
                                    try: return json.loads(match.group(1))
                                    except: pass
                                if "{" in full_content:
                                    try: return json.loads(full_content.split("}")[0] + "}")
                                    except: pass
                                return None
                        
                        elif resp.status in [429, 500, 502, 503]:
                            # Server busy -> raise to trigger a retry.
                            raise aiohttp.ClientError(f"Server Busy: {resp.status}")
                        else:
                            print(f"[Verifier API Error] Status: {resp.status}")
                            return None

            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                if attempt == max_retries - 1:
                    print(f"[Verifier Failed] {e}")
                    return None

                # Simple backoff.
                await asyncio.sleep(2)
        
        return None
        
    @classmethod
    async def close_all(cls):
        pass
        
async def main():
    # 1. Test image path (edit to a stitched image that exists on your machine).
    TEST_IMG = "/path/to/data/nuscenes/stitched_output/<sample_id>.png"

    # 2. Init
    client = VerifierClient()

    # 3. Mock data
    test_q = "What is the best way to avoid a collision? Choose from the following answers only\n                    A) Come to a complete stop.\n                    B) Brake suddenly.\n                    C) Speed up.\n                    D) Change lane to right.\n                    E) Change lane to left.\n         F) None of the options."
    test_agent_ans = "<think>\n**Step-by-Step Reasoning**:\n\n1. The front view (middle image in the top row) shows a clear road ahead with no immediate obstacles or vehicles directly in front of the ego vehicle.\n2. The front right and front left views (first and third images in the top row) show parked cars on both sides, indicating that the adjacent lanes are blocked.\n3. The bottom row images confirm the presence of parked cars and a pedestrian crossing, but no immediate threat in the current lane.\n4. Given the clear path ahead and blocked adjacent lanes, there is no immediate need to stop or change lanes.\n5. Speeding up is unnecessary as the road is clear, and there are no visible hazards requiring sudden braking.\n6. The best option is to continue at the current speed while maintaining awareness of the surroundings.\n</think>\n<answer>\n**Final Answer**: F) None of the options. The road ahead is clear, and there is no immediate need to take any evasive action.\n</answer>"
    test_gt = "D) Change lane to right."

    print("\nStarting async verifier test...")

    # 4. Run the async request
    result = await client.query(TEST_IMG, test_q, test_agent_ans, test_gt)

    # 5. Show result
    if result:
        print("\nEvaluation Result:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print("\nTest FAILED.")

    # Close the connection pool.
    await client.close_all()

if __name__ == "__main__":
    asyncio.run(main())