txt = open("inference.py", encoding="utf-8").read()

# Fix run_task signature - remove role param
txt = txt.replace(
    "async def run_task(client, task, base_url, role=\"attacker\"):",
    "async def run_task(client, task, base_url):"
)

# Fix reset call - remove role from body
txt = txt.replace(
    'json={"task": task, "seed": 42, "role": role}',
    'json={"task": task, "seed": 42}'
)

# Fix step call - remove role+last_task from body
txt = txt.replace(
    '"action_type": atype, "target_node": target, "role": role, "last_task": task},',
    '"action_type": atype, "target_node": target},'
)

# Fix the log_start - remove role suffix
txt = txt.replace(
    'log_start(task=task+"["+role+"]", model=MODEL_NAME)',
    "log_start(task=task, model=MODEL_NAME)"
)

# Fix the DEBUG line - remove role references
txt = txt.replace(
    '"[DEBUG] Role=" + role.upper() + " | task=" + task + " | defender_level=" + defender_level',
    '"[DEBUG] AI vs AI | task=" + task + " | defender_level=" + defender_level'
)

# Fix get_defender_action call - remove role reference
txt = txt.replace(
    "raw_def, def_err = get_defender_action(client, step, obs_dict, def_history, task)",
    "raw_def, def_err = get_defender_action(client, step, obs_dict, def_history, task)"
)

# Fix role check in action selection
txt = txt.replace(
    'if role == "defender":\n                    raw_att, att_err = get_defender_action(client, step, obs_dict, att_history, task)\n                else:\n                    raw_att, att_err = get_attacker_action(client, step, obs_dict, att_history)',
    "raw_att, att_err = get_attacker_action(client, step, obs_dict, att_history)"
)

# Fix default actions
txt = txt.replace(
    'default_action = "monitor" if role == "defender" else "scan"\r\n                default_node  = 3 if role == "defender" else 1\r\n                atype, target, parse_err = _parse_node(raw_att, default_action, default_node)',
    'atype, target, parse_err = _parse_node(raw_att, "scan", 1)'
)
txt = txt.replace(
    'default_action = "monitor" if role == "defender" else "scan"\n                default_node  = 3 if role == "defender" else 1\n                atype, target, parse_err = _parse_node(raw_att, default_action, default_node)',
    'atype, target, parse_err = _parse_node(raw_att, "scan", 1)'
)

# Fix is_ai_vs_ai - always True
txt = txt.replace(
    '    # ALL tasks now use TRUE AI vs AI\n    is_ai_vs_ai = True\n\n    defender_level = {"easy": "passive", "medium": "moderate", "hard": "aggressive"}.get(task, "moderate"); role_label = role.upper()',
    '    # ALL tasks: TRUE AI vs AI — LLM attacker + LLM defender in same episode\n    is_ai_vs_ai = True\n    defender_level = {"easy": "passive", "medium": "moderate", "hard": "aggressive"}.get(task, "moderate")'
)

open("inference.py", "w", encoding="utf-8").write(txt)
print("DONE")
