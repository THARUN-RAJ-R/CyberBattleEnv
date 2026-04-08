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

Tasks (ALL are TRUE AI vs AI — LLM Attacker vs LLM Defender):
  easy        - LLM Attacker vs Passive LLM Defender   (junior analyst, patch only)
  medium      - LLM Attacker vs Moderate LLM Defender  (engineer, patch+monitor+restore)
  hard        - LLM Attacker vs Aggressive LLM Defender (full arsenal, rapid response)
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
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "cyber-battle-env"

# ── Task config ────────────────────────────────────────────────────────────
TASKS = [f"level_{i}" for i in range(1, 11)]

def get_base_difficulty(task: str) -> str:
    try:
        level = int(task.split("_")[1])
    except:
        level = 1
    if level <= 3: return "easy"
    if level <= 6: return "medium"
    return "hard"

def get_max_steps(task: str) -> int:
    try: return 10 + (int(task.split("_")[1]) * 2)
    except: return 20

def get_max_reward(task: str) -> float: return 1.0
def get_success_thresh(task: str) -> float: return 0.3

TEMPERATURE = 0.4
MAX_TOKENS  = 200

# ── ATTACKER system prompts (scales with task difficulty) ─────────────────
# easy:   Junior Attacker -> 1v1 vs Junior Analyst
# medium: Advanced Attacker -> 1v1 vs Security Engineer
# hard:   Elite Nation-State Attacker -> 1v1 vs Aggressive AI Defender

_BASE_ATTACKER = """
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
AVOID: Being detected.
Reply with exactly one JSON line. Nothing else.
"""

ATTACKER_PROMPTS = {
    "easy": textwrap.dedent("""
    You are a JUNIOR script-kiddie hacker. You are attacking a junior analyst.
    Keep it simple. Scan and exploit the first server you find.
    """ + _BASE_ATTACKER).strip(),
    
    "medium": textwrap.dedent("""
    You are a PERSISTENT THREAT hacker. You are battling a skilled Security Engineer.
    Move stealthily, hop between machines, and target the database.
    """ + _BASE_ATTACKER).strip(),
    
    "hard": textwrap.dedent("""
    You are an ELITE AI NATION-STATE ATTACKER. You are battling an aggressive elite Defender AI.
    Move fast, escalate privileges, and deceive the defender. Time is against you.
    """ + _BASE_ATTACKER).strip()
}

# ── DEFENDER system prompts (scales with task difficulty) ─────────────────
# easy:   Passive LLM — only patches, never isolates. Training wheels.
# medium: Moderate LLM — patches + monitors, rare isolation.
# hard:   Aggressive LLM — full arsenal, rapid response.

DEFENDER_PROMPTS = {
    "easy": textwrap.dedent("""
    You are a JUNIOR security analyst protecting a corporate network.
    You are not very experienced, so you only do basic patching.

    YOUR ACTIONS (output ONE JSON):
      {"action_type": "patch",   "target_node": <1-3>}
      {"action_type": "monitor", "target_node": <1-3>}

    RULES:
      - Only patch or monitor. Do NOT isolate or block.
      - Patch the most vulnerable node you know about.
      - You may miss some attacker activity.

    Reply with exactly one JSON line. Nothing else.
    """).strip(),

    "medium": textwrap.dedent("""
    You are a SECURITY ENGINEER protecting a corporate network.
    An attacker is active — respond with moderate countermeasures.

    YOUR ACTIONS (output ONE JSON):
      {"action_type": "patch",    "target_node": <1-3>}
      {"action_type": "monitor",  "target_node": <1-3>}
      {"action_type": "restore",  "target_node": <1-3>}

    STRATEGY:
      - Patch whichever node the attacker just touched.
      - Monitor the Database (node 3) regularly.
      - Restore compromised nodes when possible.
      - Do NOT isolate yet — try softer measures first.

    Reply with exactly one JSON line. Nothing else.
    """).strip(),

    "hard": textwrap.dedent("""
    You are an AI DEFENDER protecting a corporate network from an active attacker.

    NETWORK:
      Node 0: User Workstation (COMPROMISED — attacker base)
      Node 1: Web Server        (adjacent to 0 and 2)
      Node 2: App Server        (adjacent to 1 and 3)
      Node 3: Database          (HIGH VALUE — protect at ALL costs)

    YOUR ACTIONS (output ONE JSON):
      {"action_type": "patch",    "target_node": <1-3>}  -> Harden node
      {"action_type": "monitor",  "target_node": <1-3>}  -> Raise detection
      {"action_type": "isolate",  "target_node": <1-3>}  -> Lock node
      {"action_type": "restore",  "target_node": <1-3>}  -> Un-compromise
      {"action_type": "block",    "target_node": <0-3>}  -> Max detection

    STRATEGY:
      - Turn 1: ALWAYS monitor(3) — protect the Database immediately.
      - Patch the exact node the attacker is heading toward.
      - If attacker reaches node 2, ISOLATE node 3 instantly.
      - Restore compromised nodes to cut off attacker movement.
      - Use block on nodes the attacker keeps returning to.

    Reply with exactly one JSON line. Nothing else.
    """).strip(),
}


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


def get_attacker_action(client, step, obs_dict, history, task="hard"):
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
                {"role": "system", "content": ATTACKER_PROMPTS.get(get_base_difficulty(task), ATTACKER_PROMPTS["hard"])},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _extract_json(raw), None
    except Exception as e:
        return '{"action_type":"scan","target_node":1}', str(e)


def get_defender_action(client, step, obs_dict, def_history, task="hard"):
    """Ask LLM for defender move. Difficulty scales with task.
    easy=passive, medium=moderate, hard=aggressive.
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
    defender_prompt = DEFENDER_PROMPTS.get(get_base_difficulty(task), DEFENDER_PROMPTS["hard"])
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": defender_prompt},
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

    max_steps  = get_max_steps(task)
    max_reward = get_max_reward(task)
    threshold  = get_success_thresh(task)

    rewards     = []
    att_history = []
    def_history = []
    steps_taken = 0
    score       = 0.0
    success     = False
    last_error  = None

    # ALL tasks now use TRUE AI vs AI
    is_ai_vs_ai = True

    log_start(task=task, model=MODEL_NAME)
    defender_level = {"easy": "passive", "medium": "moderate", "hard": "aggressive"}.get(task, "moderate")
    print("[DEBUG] AI vs AI | task=" + task + " | defender_level=" + defender_level, flush=True)

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
                raw_att, att_err = get_attacker_action(client, step, obs_dict, att_history, task)
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
                            json={"action_type": atype, "target_node": target, "last_task": task},
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
                    raw_def, def_err = get_defender_action(client, step, obs_dict, def_history, task)
                    def_atype, def_target, _ = _parse_node(raw_def, "monitor", 3)

                    def_action_str = ('{"action_type":"' + def_atype
                                      + '","target_node":' + str(def_target) + '}')

                    try:
                        dr = await http.post(
                            "/defender_step",
                            json={"action_type": def_atype, "target_node": def_target, "last_task": task},
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
    """
    ONE episode per task. Each turn:
      LLM makes ATTACKER decision -> POST /step
      LLM (different prompt) makes DEFENDER decision -> POST /defender_step
    Both happen in the SAME game, every turn.
    TRUE simultaneous AI vs AI.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    container_id = None
    base_url = ENV_BASE_URL
    if LOCAL_IMAGE_NAME:
        container_id, base_url = await _start_docker(LOCAL_IMAGE_NAME)
    try:
        print("[DEBUG] === TRUE AI vs AI: One agent, two minds, one battlefield ===", flush=True)
        print("[DEBUG] Every turn: LLM reasons as ATTACKER + LLM reasons as DEFENDER", flush=True)
        print("[DEBUG] Same model, independent system prompts, one game per task", flush=True)
        print("", flush=True)
        all_scores = []
        import httpx as _hx
        for task in TASKS:
            score, success, steps, task_rewards = await run_task(client, task, base_url)
            all_scores.append(score)
            status_text = "PASS" if success else "FAIL"
            
            def_score = 0.0
            try:
                async with _hx.AsyncClient(timeout=5) as _ui_cx:
                    ui_req = await _ui_cx.get(base_url + "/ui_state")
                    if ui_req.status_code == 200: def_score = ui_req.json().get("defender_score", 0.0)
            except: pass

            print(f"[END] {task.upper()} | Attacker: {score:.2f} | Defender: {def_score:.2f} | Status: {status_text} | Tracker: {task_rewards}", flush=True)

            # POST result to dashboard report store
            try:
                async with _hx.AsyncClient(timeout=5) as _rc:
                    await _rc.post(base_url + "/report_task", json={
                        "task": task.upper(),
                        "attacker_score": round(score, 2),
                        "defender_score": round(def_score, 2),
                        "success": success,
                        "steps": steps,
                        "rewards": task_rewards,
                    })
            except Exception:
                pass
            print("", flush=True)
        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print("[DEBUG] ===================================================", flush=True)
        print("[DEBUG] OVERALL SCORE: " + ("%.2f" % avg) + " (" + ("%.0f" % (avg*100)) + "%)", flush=True)
        print("[DEBUG] ===================================================", flush=True)
    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], check=False, capture_output=True)
            print("[DEBUG] Container stopped", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
