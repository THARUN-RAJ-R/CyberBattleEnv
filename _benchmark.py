# -*- coding: utf-8 -*-
"""
Accuracy benchmark for CyberBattleEnv.
Tests all 3 tasks with an optimal scripted agent (not LLM)
to measure the environment's achievable score ceiling.
Then tests with a random agent to get a baseline floor.
"""
import sys, os, urllib.request, json, ssl, time, random

sys.path.insert(0, '.')

import logging
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

# ── Config ────────────────────────────────────────────────
MAX_REWARD = {'easy': 0.9, 'medium': 1.0, 'hard': 1.0}
SEEDS      = [1, 7, 42, 99, 123]

# Optimal attacker playbooks
PLAYBOOKS = {
    'easy': [
        ('scan', 1), ('exploit', 1),
    ],
    'medium': [
        ('scan', 1), ('exploit', 1), ('lateral_move', 1),
        ('scan', 2), ('exploit', 2), ('lateral_move', 2),
        ('scan', 3), ('exploit', 3), ('lateral_move', 3),
        ('exfiltrate', 3),
    ],
    'hard': [
        ('scan', 1), ('exploit', 1), ('lateral_move', 1),
        ('escalate', 1),
        ('scan', 2), ('exploit', 2), ('lateral_move', 2),
        ('escalate', 2),
        ('scan', 3), ('exploit', 3), ('lateral_move', 3),
        ('exfiltrate', 3),
    ],
}

# ── Run a single episode ──────────────────────────────────
def run_episode(task, seed, actions):
    env = CyberBattleEnvironment()
    env.reset(task=task, seed=seed)
    rewards = []
    winner  = None
    steps   = 0
    for atype, tnode in actions:
        obs = env.step(CyberBattleAction(action_type=atype, target_node=tnode))
        rewards.append(obs.reward)
        steps += 1
        if obs.done:
            winner = env.state.winner
            break
    score = min(max(sum(rewards) / MAX_REWARD[task], 0.0), 1.0)
    return score, winner, steps, rewards

def run_random_episode(task, seed):
    random.seed(seed)
    env = CyberBattleEnvironment()
    env.reset(task=task, seed=seed)
    actions = ['scan', 'exploit', 'lateral_move', 'escalate', 'exfiltrate']
    rewards = []
    for _ in range(MAX_STEPS):
        atype = random.choice(actions)
        tnode = random.choice([0, 1, 2, 3])
        obs = env.step(CyberBattleAction(action_type=atype, target_node=tnode))
        rewards.append(obs.reward)
        if obs.done:
            break
    score = min(max(sum(rewards) / MAX_REWARD[task], 0.0), 1.0)
    return score

MAX_STEPS = 15

# ── Run all benchmarks ────────────────────────────────────
print('=' * 58)
print('  CYBERBATTLEENV ACCURACY BENCHMARK')
print('  %d seeds per task x 3 tasks = %d episodes each' % (len(SEEDS), len(SEEDS)))
print('=' * 58)

print()
print('--- OPTIMAL AGENT (upper bound) ---')
print()

task_results = {}
for task in ['easy', 'medium', 'hard']:
    scores = []
    wins   = 0
    for seed in SEEDS:
        score, winner, steps, rewards = run_episode(task, seed, PLAYBOOKS[task])
        scores.append(score)
        if winner == 'attacker':
            wins += 1

    avg   = sum(scores) / len(scores)
    best  = max(scores)
    worst = min(scores)
    win_r = wins / len(SEEDS) * 100
    task_results[task] = {'avg': avg, 'best': best, 'worst': worst, 'win_rate': win_r}

    print('Task: %-8s  avg=%.3f  best=%.3f  worst=%.3f  win_rate=%.0f%%' % (
        task, avg, best, worst, win_r))

overall_opt = sum(r['avg'] for r in task_results.values()) / 3
print()
print('>>> OPTIMAL AGENT OVERALL SCORE: %.3f (%.1f%%)' % (overall_opt, overall_opt * 100))

print()
print('--- RANDOM AGENT (lower bound) ---')
print()

rand_results = {}
for task in ['easy', 'medium', 'hard']:
    scores = [run_random_episode(task, s) for s in SEEDS]
    avg    = sum(scores) / len(scores)
    rand_results[task] = avg
    print('Task: %-8s  avg=%.3f (%.1f%%)' % (task, avg, avg * 100))

overall_rand = sum(rand_results.values()) / 3
print()
print('>>> RANDOM AGENT OVERALL SCORE:  %.3f (%.1f%%)' % (overall_rand, overall_rand * 100))

print()
print('--- INFERENCE RUN (LLM agent from last run) ---')
# From the last inference.py run we captured these scores
llm_scores = {'easy': 1.000, 'medium': 0.400, 'hard': 0.660}
llm_avg    = sum(llm_scores.values()) / 3
for task, s in llm_scores.items():
    print('Task: %-8s  score=%.3f (%.1f%%)' % (task, s, s * 100))
print()
print('>>> LLM AGENT OVERALL SCORE:     %.3f (%.1f%%)' % (llm_avg, llm_avg * 100))

print()
print('=' * 58)
print('  SUMMARY')
print('=' * 58)
print('  Random agent  (floor):   %.3f = %.1f%%' % (overall_rand, overall_rand * 100))
print('  LLM agent     (yours):   %.3f = %.1f%%' % (llm_avg,      llm_avg * 100))
print('  Optimal agent (ceiling): %.3f = %.1f%%' % (overall_opt,  overall_opt * 100))
print()
pct_of_optimal = (llm_avg / overall_opt * 100) if overall_opt > 0 else 0
above_random   = llm_avg - overall_rand
print('  LLM is %.1f%% of optimal' % pct_of_optimal)
print('  LLM is +%.3f above random' % above_random)
print()

# Per-task breakdown
print('  PER-TASK SCORES (LLM):')
labels = {'easy': ':easy:', 'medium': ':medium:', 'hard': ':hard:'}
thresholds = {'easy': 0.5, 'medium': 0.4, 'hard': 0.3}
for task in ['easy', 'medium', 'hard']:
    s = llm_scores[task]
    passed = s >= thresholds[task]
    bar = '#' * int(s * 20)
    print('    %-8s [%-20s] %.3f  %s' % (
        task, bar, s, 'PASS' if passed else 'FAIL (threshold=%.1f)' % thresholds[task]))

print()
tasks_passed = sum(1 for t,s in llm_scores.items() if s >= thresholds[t])
print('  Tasks above threshold: %d/3' % tasks_passed)
print('  Overall accuracy:      %.1f%%' % (llm_avg * 100))
print('=' * 58)
