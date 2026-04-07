txt = open("inference.py", encoding="utf-8").read()
# Find the main function start
idx = txt.find("# __ Main __")
if idx == -1:
    idx = txt.find("async def main():")
    # Go back to find the comment before it
    comment_idx = txt.rfind("\n\n", 0, idx)
    idx = comment_idx + 2

new_main = '''# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    """
    ONE episode per task. Each turn:
      LLM makes ATTACKER decision -> POST /step
      LLM (different prompt) makes DEFENDER decision -> POST /defender_step
    Both happen in the SAME game, every turn.
    TRUE simultaneous AI vs AI.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    container_id = None
    base_url = ENV_BASE_URL
    if IMAGE_NAME:
        container_id, base_url = await _start_docker(IMAGE_NAME)
    try:
        print("[DEBUG] === TRUE AI vs AI: One agent, two minds, one battlefield ===", flush=True)
        print("[DEBUG] Every turn: LLM reasons as ATTACKER + LLM reasons as DEFENDER", flush=True)
        print("[DEBUG] Same model, independent system prompts, one game per task", flush=True)
        print("", flush=True)
        all_scores = []
        for task in TASKS:
            score, success, steps, task_rewards = await run_task(client, task, base_url)
            all_scores.append(score)
            print("[DEBUG] task=" + task + " score=" + ("%.2f" % score) + " success=" + str(success) + " steps=" + str(steps), flush=True)
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
'''

# Replace from "# __ Main" to end of file
idx2 = txt.find("# \u2500\u2500 Main \u2500")
if idx2 == -1:
    idx2 = txt.rfind("async def main():")
    idx2 = txt.rfind("\n\n", 0, idx2) + 2

txt2 = txt[:idx2] + new_main
open("inference.py", "w", encoding="utf-8").write(txt2)
print("DONE - main() replaced. Total lines:", len(txt2.split("\n")))
