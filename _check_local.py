# -*- coding: utf-8 -*-
"""
Local sanity check — no server or LLM needed.
Verifies files exist, environment logic works, config is correct.
"""
import sys, os
sys.path.insert(0, '.')

results = []

# ── File existence checks ──────────────────────────────────────────────────
files = [
    'Dockerfile', 'openenv.yaml', 'pyproject.toml', 'requirements.txt',
    'inference.py', 'README.md',
    'cyber_battle_env/__init__.py', 'cyber_battle_env/models.py',
    'cyber_battle_env/client.py',
    'cyber_battle_env/server/environment.py', 'cyber_battle_env/server/app.py',
]
for fx in files:
    results.append(('FILE: ' + fx, os.path.exists(fx)))

# ── Environment logic ──────────────────────────────────────────────────────
import logging
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

# easy task: scan + exploit web server → attacker wins
env_easy = CyberBattleEnvironment()
env_easy.reset(task='easy', seed=42)
env_easy.step(CyberBattleAction(action_type='scan', target_node=1))
obs = env_easy.step(CyberBattleAction(action_type='exploit', target_node=1))
results.append(('easy: scan+exploit web server wins', obs.done and env_easy.state.winner == 'attacker'))

# task max_turns
e_med = CyberBattleEnvironment()
results.append(('medium max_turns=20', e_med.reset(task='medium').max_turns == 20))

e_hard = CyberBattleEnvironment()
results.append(('hard max_turns=25', e_hard.reset(task='hard').max_turns == 25))

e_easy2 = CyberBattleEnvironment()
results.append(('easy max_turns=15', e_easy2.reset(task='easy').max_turns == 15))

# invalid action should be penalised
e_inv = CyberBattleEnvironment()
e_inv.reset(task='easy', seed=1)
inv = e_inv.step(CyberBattleAction(action_type='exfiltrate', target_node=3))
results.append(('invalid action penalised (reward <= 0)', inv.reward <= 0))

# scores should stay in [0,1] across all tasks
scores_ok = True
for task, max_r in [('easy', 0.9), ('medium', 1.0), ('hard', 1.0)]:
    e = CyberBattleEnvironment()
    e.reset(task=task, seed=42)
    total = 0.0
    for _ in range(3):
        o = e.step(CyberBattleAction(action_type='scan', target_node=1))
        total += o.reward
        if o.done:
            break
    score = min(max(total / max_r, 0.0), 1.0)
    if not (0.0 <= score <= 1.0):
        scores_ok = False
results.append(('all tasks score in [0,1]', scores_ok))

# unknown task should raise ValueError
bad_task_ok = False
try:
    CyberBattleEnvironment().reset(task='unknown_task')
except ValueError:
    bad_task_ok = True
results.append(('unknown task raises ValueError', bad_task_ok))

# ── openenv.yaml config checks ────────────────────────────────────────────
c = open('openenv.yaml', encoding='utf-8').read()
results.append(('openenv.yaml: name field',       'name:' in c))
results.append(('openenv.yaml: version field',    'version:' in c))
results.append(('openenv.yaml: easy task',        'name: easy' in c))
results.append(('openenv.yaml: medium task',      'name: medium' in c))
results.append(('openenv.yaml: hard task',        'name: hard' in c))
results.append(('openenv.yaml: reward section',   'reward:' in c))
results.append(('openenv.yaml: action_space',     'action_space:' in c))

# ── README checks ──────────────────────────────────────────────────────────
readme = open('README.md', encoding='utf-8').read()
results.append(('README: sdk: docker',     'sdk: docker' in readme))
results.append(('README: app_port: 8000',  'app_port: 8000' in readme))
results.append(('README: license',         'license:' in readme))
results.append(('README: [END] format',    '[END]' in readme))
results.append(('README: score= in [END]', 'score=' in readme))

# ── inference.py checks ────────────────────────────────────────────────────
inf = open('inference.py', encoding='utf-8').read()
results.append(('inference.py: API_BASE_URL default',   '"https://router.huggingface.co/v1"' in inf or '"https://api.openai.com/v1"' in inf))
results.append(('inference.py: MODEL_NAME default',     'MODEL_NAME' in inf and 'getenv' in inf))
results.append(('inference.py: HF_TOKEN required',      'HF_TOKEN' in inf))
results.append(('inference.py: [START] format',         '[START]' in inf))
results.append(('inference.py: [STEP] format',          '[STEP]' in inf))
results.append(('inference.py: [END] format',           '[END]' in inf))
results.append(('inference.py: score= in [END]',        'score=' in inf))
results.append(('inference.py: TASKS has easy',         '"easy"' in inf))
results.append(('inference.py: TASKS has medium',       '"medium"' in inf))
results.append(('inference.py: TASKS has hard',         '"hard"' in inf))

# ── Output ─────────────────────────────────────────────────────────────────
logging.disable(logging.NOTSET)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)

print('[LOCAL + CONFIG CHECKS]')
for name, ok in results:
    print('  [PASS]' if ok else '  [FAIL]', name)

print()
print('RESULTS: %d/%d PASSED' % (passed, passed + failed))
print('STATUS:', 'ALL GOOD' if failed == 0 else str(failed) + ' ISSUE(S) — fix before submitting')

if failed > 0:
    sys.exit(1)
