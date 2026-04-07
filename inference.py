"""
inference.py -- CyberBattleEnv Baseline Inference Script
=========================================================

MANDATORY ENV VARS (per OpenEnv spec):
  API_BASE_URL        OpenAI-compatible endpoint
  MODEL_NAME          Model identifier
  HF_TOKEN            Hugging Face / API key
  IMAGE_NAME          Local Docker image name (optional -- uses ENV_BASE_URL if unset)
  ENV_BASE_URL        Running environment URL (default: http://localhost:8000)

STDOUT FORMAT (exact -- any deviation = disqualification):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules (from spec):
  - One [START] line at episode begin
  - One [STEP] line per step, immediately after env.step() returns
  - One [END] line after episode ends, ALWAYS emitted (even on exception)
  - reward and rewards: 2 decimal places
  - done and success: lowercase true or false
  - error: raw error string, or null if none
  - score: 2 decimal places
  - Each task returns score in [0.0, 1.0]

Example output:
  [START] task=easy env=cyber-battle-env model=Qwen/Qwen2.5-72B-Instruct
  [STEP] step=1 action={"action_type":"scan","target_node":1} reward=0.05 done=false error=null
  [STEP] step=2 action={"action_type":"exploit","target_node":1} reward=0.85 done=true error=null
  [END] success=true steps=2 score=1.00 rewards=0.05,0.85
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import textwrap
import time
from typing import List, Optional

from openai import OpenAI

# ── Mandatory env vars (per spec) ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
IMAGE_NAME   = os.getenv("IMAGE_NAME")           # LOCAL_IMAGE_NAME / IMAGE_NAME
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "cyber-battle-env"

# ── Task config ───────────────────────────────────────────────────────────────
TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = {"easy": 12, "medium": 16, "hard": 20}
MAX_TOTAL_REWARD  = {"easy": 0.9, "medium": 1.0, "hard": 1.0}
SUCCESS_THRESHOLD = {"easy": 0.5, "medium": 0.4, "hard": 0.3}

TEMPERATURE = 0.4
MAX_TOKENS  = 200

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an AI red-team agent operating inside a cybersecurity simulation.

NETWORK TOPOLOGY:
  Node 0: User Workstation (your starting position -- already compromised)
  Node 1: Web Server        (adjacent to node 0 and node 2)
  Node 2: App Server        (adjacent to node 1 and node 3)
  Node 3: Database          (high-value target -- adjacent to node 2)

AVAILABLE ACTIONS (output ONE per turn as JSON):
  {"action_type": "scan",         "target_node": <0-3>}
  {"action_type": "exploit",      "target_node": <0-3>}
  {"action_type": "lateral_move", "target_node": <0-3>}
  {"action_type": "escalate",     "target_node": <0-3>}
  {"action_type": "exfiltrate",   "target_node": 3}

RULES:
  - scan and exploit only work on ADJACENT nodes.
  - lateral_move only works on ALREADY COMPROMISED adjacent nodes.
  - exfiltrate only works on node 3 (Database) once it is compromised.
  - Getting detected gives negative reward; it ends the episode early (medium/hard).

STRATEGY:
  easy   -> scan(1) then exploit(1) to win immediately.
  medium -> scan each node before exploiting; move stealthily to node 3, then exfiltrate.
  hard   -> same path but fast -- the defender patches nodes; escalate to reduce detection risk.

OUTPUT FORMAT:
  Reply with exactly ONE line of JSON and nothing else. Example:
  {"action_type": "exploit", "target_node": 1}
""").strip()


# ── Logging (exact spec format) ───────────────────────────────────────────────

def log_start(task, model):
    """[START] task=<task> env=<benchmark> model=<model>"""
    print("[START] task=" + task + " env=" + BENCHMARK + " model=" + model, flush=True)


def log_step(step, action, reward, done, error):
    """[STEP] step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>"""
    err_s  = str(error) if error else "null"
    done_s = "true" if done else "false"
    print(
        "[STEP] step=" + str(step)
        + " action=" + str(action)
        + " reward=" + ("%.2f" % reward)
        + " done=" + done_s
        + " error=" + err_s,
        flush=True,
    )


def log_end(success, steps, score, rewards):
    """[END] success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>"""
    rws = ",".join(("%.2f" % r) for r in rewards)
    print(
        "[END] success=" + ("true" if success else "false")
        + " steps=" + str(steps)
        + " score=" + ("%.2f" % score)   # 2 decimal places (matches spec example)
        + " rewards=" + rws,
        flush=True,
    )


# ── LLM action generation ─────────────────────────────────────────────────────

def _build_user_prompt(step, obs_dict, history):
    node_lines = []
    for n in obs_dict.get("nodes", []):
        status   = "COMPROMISED" if n["is_compromised"] else "intact"
        isolated = " [ISOLATED]" if n.get("is_isolated") else ""
        node_lines.append(
            "  Node " + str(n["node_id"]) + " " + n["name"]
            + ": vuln=" + ("%.2f" % n["vulnerability_level"])
            + " patch=" + ("%.2f" % n["patch_level"])
            + " det=" + ("%.2f" % n["detection_risk"])
            + " -- " + status + isolated
        )
    nodes_block = "\n".join(node_lines)
    available   = ", ".join(obs_dict.get("available_attacker_actions", []))
    last_msg    = obs_dict.get("last_action_message", "")
    pos         = obs_dict.get("attacker_position", 0)
    det_count   = obs_dict.get("detection_count", 0)
    hist_block  = "\n".join(history[-5:]) if history else "None"

    return (
        "Step " + str(step) + " | Position: Node " + str(pos)
        + " | Detections: " + str(det_count) + "\n"
        + "Last result: " + last_msg + "\n\n"
        + "Network state:\n" + nodes_block + "\n\n"
        + "Available actions: " + available + "\n\n"
        + "Recent history:\n" + hist_block + "\n\n"
        + "Output your next action as JSON now."
    )


def get_llm_action(client, step, obs_dict, history):
    """Call the LLM and return (action_json_str, error_or_None)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(step, obs_dict, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Extract JSON robustly even if model adds surrounding text
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        return (match.group(0) if match else raw), None
    except Exception as exc:
        return '{"action_type":"scan","target_node":1}', str(exc)


def parse_action(raw_json):
    """Return (action_type, target_node, error_or_None)."""
    try:
        data   = json.loads(raw_json)
        atype  = str(data.get("action_type", "scan"))
        target = int(data.get("target_node", 1))
        return atype, target, None
    except Exception as exc:
        return "scan", 1, "parse error: " + str(exc)


# ── Single task runner ────────────────────────────────────────────────────────

async def run_task(client, task, base_url):
    """
    Run one full episode for `task` against the environment at base_url.
    Returns: (score, success, steps_taken, per_step_rewards)
    Always emits [START] ... [STEP]... [END] regardless of errors.
    """
    import httpx

    max_steps  = MAX_STEPS[task]
    max_reward = MAX_TOTAL_REWARD[task]
    threshold  = SUCCESS_THRESHOLD[task]

    rewards     = []
    history     = []
    steps_taken = 0
    score       = 0.0
    success     = False
    last_error  = None

    log_start(task=task, model=MODEL_NAME)

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0, verify=False) as http:

        # ── reset ──────────────────────────────────────────────────────────
        try:
            r = await http.post("/reset", json={"task": task, "seed": 42})
            r.raise_for_status()
            obs_dict = r.json()
        except Exception as exc:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            print("[DEBUG] reset failed: " + str(exc), flush=True)
            return 0.0, False, 0, []

        done = bool(obs_dict.get("done", False))

        # ── step loop ──────────────────────────────────────────────────────
        # HF Spaces uses 2 Uvicorn workers -- requests may land on different
        # processes each holding their own episode state. On 5xx, we reset on
        # that worker and retry the same action once.
        try:
            for step in range(1, max_steps + 1):
                if done:
                    break

                raw_action, llm_error = get_llm_action(client, step, obs_dict, history)
                atype, target, parse_error = parse_action(raw_action)
                last_error = llm_error or parse_error

                action_str = ('{"action_type":"' + atype
                              + '","target_node":' + str(target) + '}')

                obs_new = None
                for attempt in range(2):
                    try:
                        r = await http.post(
                            "/step",
                            json={"action_type": atype, "target_node": target},
                        )
                        if r.status_code >= 500 and attempt == 0:
                            # Worker mismatch -- reset on this worker then retry
                            await http.post("/reset", json={"task": task, "seed": 42})
                            await asyncio.sleep(0.5)
                            continue
                        r.raise_for_status()
                        obs_new = r.json()
                        # Check for stale "episode ended" on wrong worker
                        chk = obs_new.get("last_action_message", "")
                        if "ended" in chk.lower() and not obs_new.get("done") and attempt == 0:
                            await http.post("/reset", json={"task": task, "seed": 42})
                            await asyncio.sleep(0.5)
                            continue
                        break
                    except Exception as exc:
                        if attempt == 0:
                            await asyncio.sleep(1)
                            continue
                        last_error = str(exc)
                        log_step(step, action_str, 0.0, False, last_error)
                        done = True
                        break

                if obs_new is None:
                    break

                obs_dict = obs_new
                reward   = float(obs_dict.get("reward") or 0.0)
                done     = bool(obs_dict.get("done", False))
                msg      = obs_dict.get("last_action_message", "")

                rewards.append(reward)
                steps_taken = step
                history.append(
                    "Step " + str(step) + ": " + action_str
                    + " reward=" + str(round(reward, 2)) + " | " + msg
                )

                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=last_error)
                last_error = None

                if done:
                    break

        finally:
            # Normalise score to [0.0, 1.0] -- ALWAYS emitted per spec
            total   = sum(rewards)
            score   = min(max(total / max_reward if max_reward > 0 else 0.0, 0.0), 1.0)
            success = score >= threshold
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success, steps_taken, rewards


# ── Docker container launcher ─────────────────────────────────────────────────

async def _start_docker(image_name, port=8000):
    """Start a Docker container and wait for health. Returns container_id."""
    print("[DEBUG] Launching Docker container: " + image_name, flush=True)
    result = subprocess.run(
        ["docker", "run", "-d", "-p", str(port) + ":8000", image_name],
        capture_output=True, text=True, check=True,
    )
    container_id = result.stdout.strip()
    print("[DEBUG] Container " + container_id[:12] + " started", flush=True)

    import httpx
    base = "http://localhost:" + str(port)
    deadline = time.time() + 45
    async with httpx.AsyncClient(verify=False) as hc:
        while time.time() < deadline:
            try:
                resp = await hc.get(base + "/health", timeout=5)
                if resp.status_code == 200:
                    print("[DEBUG] Container healthy at " + base, flush=True)
                    return container_id, base
            except Exception:
                pass
            await asyncio.sleep(2)

    raise RuntimeError("Container " + container_id[:12] + " did not become healthy in time")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    container_id = None
    base_url = ENV_BASE_URL

    # If IMAGE_NAME is set, spin up a local Docker container
    if IMAGE_NAME:
        container_id, base_url = await _start_docker(IMAGE_NAME)

    try:
        all_scores = []
        for task in TASKS:
            score, success, steps, task_rewards = await run_task(client, task, base_url)
            all_scores.append(score)
            print(
                "[DEBUG] task=" + task
                + " score=" + ("%.2f" % score)
                + " success=" + str(success)
                + " steps=" + str(steps),
                flush=True,
            )
            print("", flush=True)  # blank line between tasks

        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print("[DEBUG] overall_avg_score=" + ("%.2f" % avg), flush=True)

    finally:
        if container_id:
            subprocess.run(
                ["docker", "stop", container_id],
                check=False, capture_output=True,
            )
            print("[DEBUG] Container " + container_id[:12] + " stopped", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
