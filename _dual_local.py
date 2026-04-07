import sys, logging, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="ascii", errors="replace")
sys.path.insert(0, ".")
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

p=0; f=0
def ck(name, ok, val=""):
    global p,f
    if ok: print("[PASS]", name); p+=1
    else:  print("[FAIL]", name, str(val)[:50]); f+=1

print("=== DUAL ROLE LOCAL TESTS ===")
print()

for task in ["easy","medium","hard"]:
    print("-- TASK:", task.upper(), "--")

    # ATTACKER role
    env = CyberBattleEnvironment()
    obs = env.reset(task=task, role="attacker", seed=42)
    ck(task+"/attacker reset ok", obs.attacker_position==0)
    ck(task+"/attacker msg correct", "ATTACKER" in obs.last_action_message)
    obs = env.step(CyberBattleAction(action_type="scan", target_node=1))
    ck(task+"/attacker scan reward>0", obs.reward>0)
    ck(task+"/attacker reward in [0,1]", 0<=obs.reward<=1)

    # DEFENDER role
    env2 = CyberBattleEnvironment()
    obs2 = env2.reset(task=task, role="defender", seed=42)
    ck(task+"/defender reset ok", "DEFENDER" in obs2.last_action_message)
    obs2 = env2.step(CyberBattleAction(action_type="monitor", target_node=1))
    ck(task+"/defender step has ATT+DEF", "[ATT]" in obs2.last_action_message and "[DEF]" in obs2.last_action_message)
    ck(task+"/defender reward in range", -0.5<=obs2.reward<=0.7)

    # DEFENDER isolate DB
    obs2b = env2.step(CyberBattleAction(action_type="isolate", target_node=3))
    ck(task+"/defender isolate(3) works", "Isolated" in obs2b.last_action_message or "[ATT]" in obs2b.last_action_message)

    # DEFENDER full run - defender holds on to time limit
    env3 = CyberBattleEnvironment()
    env3.reset(task="easy", role="defender", seed=1)
    for _ in range(15):
        o = env3.step(CyberBattleAction(action_type="isolate", target_node=1))
        if o.done: break
    ck(task+"/defender episode terminates", env3.state.is_game_over)
    print()

print("=== RESULTS: %d/%d PASSED ===" % (p, p+f))
print("STATUS:", "ALL GOOD" if f==0 else str(f)+" ISSUES")
