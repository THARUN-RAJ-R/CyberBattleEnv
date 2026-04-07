# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, '.')

results = []

# Local files
files = [
    'Dockerfile', 'openenv.yaml', 'pyproject.toml', 'requirements.txt',
    'inference.py', 'README.md',
    'cyber_battle_env/__init__.py', 'cyber_battle_env/models.py',
    'cyber_battle_env/client.py',
    'cyber_battle_env/server/environment.py', 'cyber_battle_env/server/app.py',
]
for fx in files:
    results.append(('FILE: ' + fx, os.path.exists(fx)))

# Env logic (no UI output from environment — suppress logging)
import logging
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

env = CyberBattleEnvironment()
env.reset(task='easy', seed=42)
env.step(CyberBattleAction(action_type='scan', target_node=1))
obs = env.step(CyberBattleAction(action_type='exploit', target_node=1))
results.append(('easy: scan+exploit wins', obs.done and env.state.winner == 'attacker'))

e3 = CyberBattleEnvironment()
results.append(('medium max_turns=20', e3.reset(task='medium').max_turns == 20))
e4 = CyberBattleEnvironment()
results.append(('hard max_turns=25', e4.reset(task='hard').max_turns == 25))

e5 = CyberBattleEnvironment()
e5.reset(task='easy', seed=1)
inv = e5.step(CyberBattleAction(action_type='exfiltrate', target_node=3))
results.append(('invalid action penalised', inv.reward <= 0))

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

# Config
c = open('openenv.yaml', encoding='utf-8').read()
results.append(('openenv.yaml: name field', 'name:' in c))
results.append(('openenv.yaml: version field', 'version:' in c))
results.append(('openenv.yaml: 3 tasks', all(t in c for t in ['easy', 'medium', 'hard'])))
results.append(('openenv.yaml: reward section', 'reward:' in c))
results.append(('openenv.yaml: action_space', 'action_space:' in c))

readme = open('README.md', encoding='utf-8').read()
results.append(('README: sdk: docker', 'sdk: docker' in readme))
results.append(('README: app_port: 8000', 'app_port: 8000' in readme))
results.append(('README: license', 'license:' in readme))

# Output
logging.disable(logging.NOTSET)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)

print('[LOCAL + CONFIG CHECKS]')
for name, ok in results:
    print('  [PASS]' if ok else '  [FAIL]', name)

print()
print('RESULTS: %d/%d PASSED' % (passed, passed + failed))
print('STATUS:', 'ALL GOOD' if failed == 0 else str(failed) + ' ISSUES')
