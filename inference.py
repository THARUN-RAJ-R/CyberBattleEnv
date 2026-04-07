"""
inference.py -- CyberBattleEnv Baseline Inference Script
=========================================================

MANDATORY ENV VARS (per OpenEnv spec):
  API_BASE_URL        OpenAI-compatible endpoint
  MODEL_NAME          Model identifier
  HF_TOKEN            Hugging Face / API key
  IMAGE_NAME          Local Docker image name (optional)
  ENV_BASE_URL        Running environment URL (default: http://localhost:8000)

STDOUT FORMAT (exact):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Tasks:
  easy        - Attacker only vs passive environment
  medium      - Attacker only vs scripted moderate defender
  hard        - Attacker + LLM Defender (TRUE AI vs AI)
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
IMAGE_NAME   = os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "cyber-battle-env"

# ── Task config ────────────────────────────────────────────────────────────
TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = {"easy": 12, "medium": 16, "hard": 20}
MAX_TOTAL_REWARD  = {"easy": 0.9, "medium": 1.0, "hard": 1.0}
SUCCESS_THRESHOLD = {"easy": 0.5, "medium": 0.4, "hard": 0.3}

TEMPERATURE = 0.4
MAX_TOKENS  = 200

# ── ATTACKER system prompt ────────────────────────────────────────────────
ATTACKER_PROMPT = textwrap.dedent("""
You are an AI red-team ATTACKER in a cybersecurity simulation.

NETWORK:
  Node 0: User Workstation (start — already compromised)
  Node 1: Web Server        (adjacent to 0 and 2)
  Node 2: App Server        (adjacent to 1 and 3)
  Node 3: Database          (final target — adjacent to 2)

YOUR ACTIONS (output ONE JSON):
  {"action_type": "scan",         "target_node": <0-3>}
  {"action_type": "exploit",      "target_node": <0-3>}
  {"action_type": "lateral_move", "target_node": <0-3>}
  {"action_type": "escalate",     "target_node": <0-3>}
  {"action_type": "exfiltrate",   "target_node": 3}

GOAL: Compromise all nodes and exfiltrate the database.
AVOID: Being detected (costs reward, ends episode on hard task).

STRATEGY:
  easy   -> scan(1), exploit(1)
  medium -> move stealthily from node 0 to node 3, then exfiltrate
  hard   -> move fast! An AI defender is actively working against you.
            Use scan to identify low-risk paths, escalate to reduce
            detection, and exfiltrate before the defender patches everything.

Reply with exactly one JSON line. Nothing else.
""").strip()

# ── DEFENDER system prompt (for hard task — TRUE AI vs AI) ────────────────
DEFENDER_PROMPT = textwrap.dedent("""
You are an AI DEFENDER protecting a corporate network from an active attacker.

NETWORK:
  Node 0: User Workstation (COMPROMISED — attacker's base)
  Node 1: Web Server        (adjacent to 0 and 2)
  Node 2: App Server        (adjacent to 1 and 3)
  Node 3: Database          (HIGH VALUE — must protect at all costs)

YOUR ACTIONS (output ONE JSON):
  {"action_type": "patch",    "target_node": <1-3>}   -> Reduce vulnerability
  {"action_type": "monitor",  "target_node": <1-3>}   -> Increase detection risk
  {"action_type": "isolate",  "target_node": <1-3>}   -> Block node from attacker
  {"action_type": "restore",  "target_node": <1-3>}   -> Un-compromise a node
  {"action_type": "block",    "target_node": <0-3>}   -> Block attacker at position

GOAL: Stop the attacker before they reach and exfiltrate the Database (node 3).
WIN: Detect the attacker 3 times OR block them from reaching the database.

STRATEGY:
  - ALWAYS protect node 3 (Database) first — monitor or isolate it
  - Patch the node the attacker is moving toward
  - If attacker is near node 2 or 3, ISOLATE node 3 immediately
  - Monitor compromised nodes to raise detection risk
  - Restore nodes the attacker has compromised to slow them down

You can see the FULL network state including attacker position.
Reply with exactly one JSON line. Nothing else.
""").strip()


# ── Logging (exact spec format) ────────────────────────────────────────────

def log_start(task, model):
    print("[START] task=" + task + " env=" + BENCHMARK + " model=" + model, flush=True)

def log_step(step, action, reward, done, error):
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
    rws = ",".join(("%.2f" % r) for r in rewards)
    print(
        "[END] success=" + ("true" if success else "false")
        + " steps=" + str(steps)
        + " score=" + ("%.2f" % score)
        + " rewards=" + rws,
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    return match.group(0) if match else raw

def _parse_node(raw_json: str, default_type: str, default_node: int):
    try:
        d = json.loads(raw_json)
        return str(d.get("action_type", default_type)), int(d.get("target_node", default_node)), None
    except Exception as e:
        return default_type, default_node, str(e)

def _build_network_block(obs_dict: dict) -> str:
    lines = []
    for n in obs_dict.get("nodes", []):
        status   = "COMPROMISED" if n["is_compromised"] else "intact"
        isolated = " [ISOLATED]"  if n.get("is_isolated")  else ""
        lines.append(
            "  Node " + str(n["node_id"]) + " " + n["name"]
            + ": vuln=" + ("%.2f" % n["vulnerability_level"])
            + " patch=" + ("%.2f" % n["patch_level"])
            + " det=" + ("%.2f" % n["detection_risk"])
            + " — " + status + isolated
        )
    return "\n".join(lines)


def get_attacker_action(client, step, obs_dict, history):
    """Ask LLM for attacker move. Returns (action_str, error)."""
    net   = _build_network_block(obs_dict)
    avail = ", ".join(obs_dict.get("available_attacker_actions", []))
    pos   = obs_dict.get("attacker_position", 0)
    det   = obs_dict.get("detection_count", 0)
    hist  = "\n".join(history[-5:]) if history else "None"
    msg   = obs_dict.get("last_action_message", "")

    prompt = (
        "Step " + str(step) + " | Your position: Node " + str(pos)
        + " | Times detected: " + str(det) + "\n"
        + "Last result: " + msg + "\n\n"
        + "Network:\n" + net + "\n\n"
        + "Available actions: " + avail + "\n\n"
        + "History:\n" + hist + "\n\n"
        + "Your next action (JSON):"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": ATTACKER_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _extract_json(raw), None
    except Exception as e:
        return '{"action_type":"scan","target_node":1}', str(e)


def get_defender_action(client, step, obs_dict, def_history):
    """Ask LLM for defender move. Returns (action_str, error).
    Only used in the 'hard' task (TRUE AI vs AI mode).
    """
    net       = _build_network_block(obs_dict)
    att_pos   = obs_dict.get("attacker_position", 0)
    det_count = obs_dict.get("detection_count", 0)
    comp      = obs_dict.get("compromised_nodes", [])
    hist      = "\n".join(def_history[-5:]) if def_history else "None"

    prompt = (
        "Step " + str(step) + " | Attacker position: Node " + str(att_pos)
        + " | Attacker detections: " + str(det_count) + "\n"
        + "Compromised nodes: " + str(comp) + "\n\n"
        + "Network status:\n" + net + "\n\n"
        + "Your defense history:\n" + hist + "\n\n"
        + "Your next defensive action (JSON):"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": DEFENDER_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _extract_json(raw), None
    except Exception as e:
        return '{"action_type":"monitor","target_node":3}', str(e)


# ── Single task runner ────────────────────────────────────────────────────

async def run_task(client, task, base_url):
    """
    Run one full episode.
    - easy/medium: LLM Attacker vs scripted defender (in environment)
    - hard:        LLM Attacker vs LLM Defender (TRUE AI vs AI)
                   Defender action is sent via POST /defender_step
    """
    import httpx

    max_steps  = MAX_STEPS[task]
    max_reward = MAX_TOTAL_REWARD[task]
    threshold  = SUCCESS_THRESHOLD[task]

    rewards     = []
    att_history = []
    def_history = []
    steps_taken = 0
    score       = 0.0
    success     = False
    last_error  = None

    is_ai_vs_ai = (task == "hard")

    log_start(task=task, model=MODEL_NAME)
    if is_ai_vs_ai:
        print("[DEBUG] HARD task: TRUE AI vs AI mode — LLM Attacker vs LLM Defender", flush=True)

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

        try:
            for step in range(1, max_steps + 1):
                if done:
                    break

                # ── ATTACKER turn (LLM) ────────────────────────────────
                raw_att, att_err = get_attacker_action(client, step, obs_dict, att_history)
                atype, target, parse_err = _parse_node(raw_att, "scan", 1)
                last_error = att_err or parse_err

                att_action_str = ('{"action_type":"' + atype
                                  + '","target_node":' + str(target) + '}')

                # Send attacker action with retry for HF multi-worker
                obs_new = None
                for attempt in range(2):
                    try:
                        r = await http.post(
                            "/step",
                            json={"action_type": atype, "target_node": target},
                        )
                        if r.status_code >= 500 and attempt == 0:
                            await http.post("/reset", json={"task": task, "seed": 42})
                            await asyncio.sleep(0.5)
                            continue
                        r.raise_for_status()
                        obs_new = r.json()
                        if ("ended" in obs_new.get("last_action_message","").lower()
                                and not obs_new.get("done") and attempt == 0):
                            await http.post("/reset", json={"task": task, "seed": 42})
                            await asyncio.sleep(0.5)
                            continue
                        break
                    except Exception as exc:
                        if attempt == 0:
                            await asyncio.sleep(1)
                            continue
                        last_error = str(exc)
                        log_step(step, att_action_str, 0.0, False, last_error)
                        done = True
                        break

                if obs_new is None:
                    break

                obs_dict    = obs_new
                att_reward  = float(obs_dict.get("reward") or 0.0)
                done        = bool(obs_dict.get("done", False))
                att_msg     = obs_dict.get("last_action_message", "")

                rewards.append(att_reward)
                steps_taken = step
                att_history.append(
                    "Step " + str(step) + ": " + att_action_str
                    + " reward=" + str(round(att_reward, 2)) + " | " + att_msg
                )

                log_step(step=step, action=att_action_str, reward=att_reward,
                         done=done, error=last_error)
                last_error = None

                if done:
                    break

                # ── DEFENDER turn (LLM — only on hard task) ───────────
                if is_ai_vs_ai and not done:
                    raw_def, def_err = get_defender_action(client, step, obs_dict, def_history)
                    def_atype, def_target, _ = _parse_node(raw_def, "monitor", 3)

                    def_action_str = ('{"action_type":"' + def_atype
                                      + '","target_node":' + str(def_target) + '}')

                    try:
                        dr = await http.post(
                            "/defender_step",
                            json={"action_type": def_atype, "target_node": def_target},
                        )
                        if dr.status_code == 200:
                            obs_dict = dr.json()
                            done = bool(obs_dict.get("done", False))
                            def_history.append(
                                "Step " + str(step) + " DEF: " + def_action_str
                            )
                            print("[DEBUG] Defender: " + def_action_str, flush=True)
                    except Exception as def_exc:
                        # Defender endpoint optional — fall back to scripted
                        print("[DEBUG] defender_step unavailable, using scripted: " + str(def_exc)[:60], flush=True)

        finally:
            total   = sum(rewards)
            score   = min(max(total / max_reward if max_reward > 0 else 0.0, 0.0), 1.0)
            success = score >= threshold
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success, steps_taken, rewards


# ── Docker container launcher ─────────────────────────────────────────────

async def _start_docker(image_name, port=8000):
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
                if (await hc.get(base + "/health", timeout=5)).status_code == 200:
                    print("[DEBUG] Container healthy", flush=True)
                    return container_id, base
            except Exception:
                pass
            await asyncio.sleep(2)
    raise RuntimeError("Container did not become healthy in time")


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    container_id = None
    base_url = ENV_BASE_URL

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
            print("", flush=True)

        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print("[DEBUG] overall_avg_score=" + ("%.2f" % avg), flush=True)

    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], check=False, capture_output=True)
            print("[DEBUG] Container stopped", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
