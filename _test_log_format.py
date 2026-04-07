import sys
sys.path.insert(0, '.')
from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

def log_start(task, model):
    print('[START] task=%s env=cyber-battle-env model=%s' % (task, model))

def log_step(step, action, reward, done, error):
    err_s = error if error else 'null'
    done_s = str(done).lower()
    print('[STEP] step=%d action=%s reward=%.2f done=%s error=%s' % (step, action, reward, done_s, err_s))

def log_end(success, steps, score, rewards):
    rws = ','.join('%.2f' % r for r in rewards)
    print('[END] success=%s steps=%d score=%.3f rewards=%s' % (str(success).lower(), steps, score, rws))

env = CyberBattleEnvironment()
env.reset(task='easy', seed=42)

log_start('easy', 'Qwen/Qwen2.5-72B-Instruct')
actions = [('scan', 1), ('exploit', 1)]
rewards = []
for i, (at, tn) in enumerate(actions, 1):
    obs = env.step(CyberBattleAction(action_type=at, target_node=tn))
    rewards.append(obs.reward)
    action_str = '{"action_type": "' + at + '", "target_node": ' + str(tn) + '}'
    log_step(i, action_str, obs.reward, obs.done, None)
    if obs.done:
        break

score = min(sum(rewards) / 0.9, 1.0)
log_end(score >= 0.5, len(rewards), score, rewards)
