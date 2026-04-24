"""
Inference Script - FusionOps V2 Environment
=============================================
STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio, json, os, sys, subprocess, textwrap
from typing import List, Optional

def _ensure_package(mod, pip):
    try: __import__(mod); return
    except ImportError: pass
    for extra in [[], ["--break-system-packages"]]:
        try:
            subprocess.check_call([sys.executable,"-m","pip","install","--quiet",pip]+extra, stderr=subprocess.DEVNULL)
            return
        except: pass

_ensure_package("openai","openai>=1.0.0")
_ensure_package("aiohttp","aiohttp>=3.9.0")
_ensure_package("pydantic","pydantic>=2.0.0")

from openai import OpenAI
from fusionops_env import FusionOpsAction, FusionOpsEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "fusionops"
TASKS = ["task1_chain","task2_residual","task3_attention","task4_mixed","task5_adversarial"]
MAX_STEPS = {"task1_chain":8,"task2_residual":12,"task3_attention":16,"task4_mixed":24,"task5_adversarial":20}
TEMPERATURE = 0.3; MAX_TOKENS = 200; SUCCESS_THRESHOLD = 0.05

SYSTEM_PROMPT = textwrap.dedent("""
You are an RL agent scheduling ML computation graph operations.

You receive a JSON observation with:
- nodes: the computation graph (first step only)
- current_node: the node you must schedule now
- current_group: ops already in the current fusion kernel
- fast_mem: node IDs whose outputs are in fast memory
- capacity: fast memory capacity in bytes
- max_fusion: max ops per fused kernel
- future_uses: how many downstream ops still need each node's output
- lookahead: next 2 nodes in the schedule

Your action is a JSON object with exactly these fields:
- fuse_with_prev: boolean (merge this op into current kernel group?)
- tile: integer (32, 64, 128, or 256)
- retain: list of node IDs to keep in fast memory

Strategy:
- Fuse consecutive connected ops to eliminate intermediate memory transfers
- Retain outputs that have future_uses > 0 and will be needed soon
- Use smaller tiles when memory is tight
- Don't retain tensors with 0 future uses (wastes memory)

RESPOND WITH ONLY A JSON OBJECT. No explanation, no markdown.
Example: {"fuse_with_prev": true, "tile": 128, "retain": [3]}
""").strip()

def log_start(task,env,model): print(f"[START] task={task} env={env} model={model}",flush=True)
def log_step(step,action,reward,done,error):
    print(f"[STEP] step={step} action={action.replace(chr(10),' ').strip()} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",flush=True)
def log_end(success,steps,score,rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}",flush=True)

def get_model_action(client, observation, history):
    hist_block = "\n".join(history[-4:]) if history else ""
    prompt = f"Observation:\n{observation}"
    if hist_block: prompt += f"\n\nPrevious actions:\n{hist_block}"
    prompt += "\n\nYour action (JSON only):"
    try:
        c = client.chat.completions.create(model=MODEL_NAME,messages=[
            {"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            temperature=TEMPERATURE,max_tokens=MAX_TOKENS,stream=False)
        text = (c.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
        try: json.loads(text)
        except: text = '{"fuse_with_prev":false,"tile":128,"retain":[]}'
        return text
    except Exception as e:
        print(f"[DEBUG] Model error: {e}",flush=True)
        return '{"fuse_with_prev":false,"tile":128,"retain":[]}'

async def run_task(client, env, task_name):
    history,rewards,steps_taken,score,success = [],[], 0, 0.0, False
    max_steps = MAX_STEPS.get(task_name,16)
    log_start(task=task_name,env=BENCHMARK,model=MODEL_NAME)
    try:
        result = await env.reset(task=task_name)
        observation = result.observation.text
        for step in range(1,max_steps+1):
            if result.done: break
            action_text = get_model_action(client,observation,history)
            result = await env.step(FusionOpsAction(command=action_text))
            rewards.append(result.reward); steps_taken = step
            observation = result.observation.text
            log_step(step=step,action=action_text,reward=result.reward,done=result.done,error=result.observation.error)
            history.append(f"Step {step}: {action_text} -> reward={result.reward:.2f}")
            if result.done:
                if result.score is not None: score = result.score
                break
        score = min(max(score,0.0),1.0); success = score >= SUCCESS_THRESHOLD
    except Exception as e: print(f"[DEBUG] Task {task_name} error: {e}",flush=True)
    finally:
        try: await env.close()
        except: pass
        log_end(success=success,steps=steps_taken,score=score,rewards=rewards)

async def main():
    try: client = OpenAI(base_url=API_BASE_URL,api_key=API_KEY)
    except Exception as e: print(f"[DEBUG] Client init failed: {e}",flush=True); return
    for task_name in TASKS:
        try:
            env = await FusionOpsEnv.from_docker_image(IMAGE_NAME)
            await run_task(client,env,task_name)
        except Exception as e:
            print(f"[DEBUG] Task {task_name} error: {e}",flush=True)
            log_end(success=False,steps=0,score=0.0,rewards=[])

if __name__ == "__main__":
    try: asyncio.run(main())
    except Exception as e: print(f"[DEBUG] Top-level error: {e}",flush=True); sys.exit(0)
