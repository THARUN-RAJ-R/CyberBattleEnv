"""
inference.py -- CyberBattleEnv Baseline Inference Script
=========================================================

MANDATORY ENV VARS:
  API_BASE_URL   OpenAI-compatible endpoint  (default: HF router)
  MODEL_NAME     Model identifier            (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Hugging Face / API key
  IMAGE_NAME     Docker image name           (optional)
  ENV_BASE_URL   Running environment URL     (default: http://localhost:8000)

STDOUT FORMAT (exact):
  [START] task=<task> env=cyber-battle-env model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
IMAGE_NAME   = os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK    = "cyber-battle-env"

# ── Task config ───────────────────────────────────────────────────────────────
TASKS            = ["easy", "medium", "hard"]
MAX_STEPS        = {"easy": 12, "medium": 16, "hard": 20}
MAX_TOTAL_REWARD = {"easy": 0.9, "medium": 1.0, "hard": 1.0}
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
  - exfiltrate only works on node 3 once compromised.
  - Detection ends the episode early (medium/hard tasks).

STRATEGY:
  easy   -> scan(1), exploit(1)
  medium -> scan(1), exploit(1), lateral_move to 1, exploit(2), lateral_move to 2, exploit(3), lateral_move to 3, exfiltrate(3)
  hard   -> same path but faster -- defender patches nodes; escalate after exploiting to reduce detection risk

OUTPUT FORMAT:
  Reply with exactly ONE line of JSON. Nothing else. Example:
  {"action_type": "exploit", "target_node": 1}
""").strip()


# ── Logging helpers (strict format) ──────────────────────────────────────────

def log_start(task, model):
    print("[START] task=" + task + " env=" + BENCHMARK + " model=" + model, flush=True)


def log_step(step, action, reward, done, error):
    err_s  = str(error) if error else "null"
    done_s = "true" if done else "false"
    print("[STEP] step=" + str(step) + " action=" + action
          + " reward=" + ("%.2f" % reward)
          + " done=" + done_s
          + " error=" + err_s, flush=True)


def log_end(success, steps, score, rewards):
    rws_s = ",".join(("%.2f" % r) for r in rewards)
    print("[END] success=" + ("true" if success else "false")
          + " steps=" + str(steps)
          + " score=" + ("%.3f" % score)
          + " rewards=" + rws_s, flush=True)


# ── LLM action generation ─────────────────────────────────────────────────────

def _build_user_prompt(step, obs_dict, history):
    node_lines = []
    for n in obs_dict.get("nodes", []):
        status   = "COMPROMISED" if n["is_compromised"] else "intact"
        isolated = " [ISOLATED]"  if n.get("is_isolated")  else ""
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
        "Step " + str(step) + " | Attacker position: Node " + str(pos)
        + " | Detections so far: " + str(det_count) + "\n"
        + "Last result: " + last_msg + "\n\n"
        + "Current network state:\n" + nodes_block + "\n\n"
        + "Available actions: " + available + "\n\n"
        + "Recent history:\n" + hist_block + "\n\n"
        + "Output your next action as JSON now."
    )


def get_llm_action(client, step, obs_dict, history):
    """Returns (action_json_string, error_or_None)."""
    user_prompt = _build_user_prompt(step, obs_dict, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            return match.group(0), None
        return raw, None
    except Exception as exc:
        return '{"action_type": "scan", "target_node": 1}', str(exc)


def parse_action(raw_json):
    """Returns (action_type, target_node, error_or_None)."""
    try:
        data   = json.loads(raw_json)
        atype  = str(data.get("action_type", "scan"))
        target = int(data.get("target_node", 1))
        return atype, target, None
    except Exception as exc:
        return "scan", 1, "JSON parse error: " + str(exc)


# ── Single task runner ────────────────────────────────────────────────────────

async def run_task(client, task, base_url):
    """
    Run one full episode for `task`.
    Returns: (score, success, steps_taken, per_step_rewards)
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

    # verify=False to handle potential SSL quirks on remote HF Spaces
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
        # HF Spaces runs 2 Uvicorn workers. Requests may hit different processes
        # each holding their own env state. On 5xx we reset on that worker and retry.
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
                            # Worker mismatch -- reset on this worker and retry
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
            total   = sum(rewards)
            score   = min(max(total / max_reward if max_reward > 0 else 0.0, 0.0), 1.0)
            success = score >= threshold
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success, steps_taken, rewards


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    container_id = None
    base_url = ENV_BASE_URL

    if IMAGE_NAME:
        import subprocess
        import time
        print("[DEBUG] Launching Docker container from " + IMAGE_NAME, flush=True)
        result = subprocess.run(
            ["docker", "run", "-d", "-p", "8000:8000", IMAGE_NAME],
            capture_output=True, text=True, check=True,
        )
        container_id = result.stdout.strip()
        print("[DEBUG] Container " + container_id[:12] + " started", flush=True)
        import httpx
        deadline = time.time() + 40
        async with httpx.AsyncClient(verify=False) as hc:
            while time.time() < deadline:
                try:
                    resp = await hc.get(base_url + "/health")
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(2)

    try:
        all_scores = []
        for task in TASKS:
            score, success, steps, task_rewards = await run_task(client, task, base_url)
            all_scores.append(score)
            print("[DEBUG] task=" + task + " score=" + ("%.3f" % score)
                  + " success=" + str(success) + " steps=" + str(steps), flush=True)
            print("", flush=True)

        avg = sum(all_scores) / len(all_scores)
        print("[DEBUG] overall_avg_score=" + ("%.3f" % avg), flush=True)

    finally:
        if container_id:
            import subprocess
            subprocess.run(["docker", "stop", container_id],
                           check=False, capture_output=True)
            print("[DEBUG] Container " + container_id[:12] + " stopped", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
