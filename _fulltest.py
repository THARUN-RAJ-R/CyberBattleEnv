import sys, io, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="ascii", errors="replace")
sys.path.insert(0, ".")
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

SEP = "=" * 70

def battle(task, role, att_moves, def_moves):
    env = CyberBattleEnvironment()
    obs = env.reset(task=task, role=role, seed=42)
    print("  Role=%-8s  Task=%-6s  %s" % (role.upper(), task.upper(), obs.last_action_message.encode("ascii","replace").decode()[:40]))
    print("  %s" % ("-"*66))
    print("  %-4s %-10s %-22s %+6s  %s" % ("TURN","SIDE","ACTION","REWARD","MSG"))
    print("  %s" % ("-"*66))

    rewards = []
    for turn in range(1, 20):
        if obs.done:
            break
        if role == "attacker":
            atype, anode = att_moves[(turn-1) % len(att_moves)]
            obs = env.step(CyberBattleAction(action_type=atype, target_node=anode))
            rewards.append(obs.reward)
            msg = obs.last_action_message.encode("ascii","replace").decode()[:32]
            print("  %-4d %-10s %-22s %+6.2f  %s" % (turn, "ATTACKER", atype+"("+str(anode)+")", obs.reward, msg))
        else:
            atype, anode = def_moves[(turn-1) % len(def_moves)]
            obs = env.step(CyberBattleAction(action_type=atype, target_node=anode))
            rewards.append(obs.reward)
            msg = obs.last_action_message.encode("ascii","replace").decode()[:32]
            print("  %-4d %-10s %-22s %+6.2f  %s" % (turn, "DEFENDER", atype+"("+str(anode)+")", obs.reward, msg))
        if obs.done:
            break

    st = env.state
    total = st.total_reward
    max_r = {"easy":0.9,"medium":1.0,"hard":1.0}[task]
    score = min(max(total/max_r, 0.0), 1.0)
    winner = st.winner or "none"
    print("  %s" % ("-"*66))
    print("  WINNER=%-8s  det=%d  turns=%d  reward=%.3f  SCORE=%.2f (%.0f%%)" % (
        winner.upper(), st.detection_count, st.step_count, total, score, score*100))
    return score

# Scripted moves for each role
ATT_EASY   = [("scan",1),("exploit",1)]
ATT_MEDIUM = [("scan",1),("exploit",1),("lateral_move",1),("scan",2),("exploit",2),("lateral_move",2),("scan",3),("exploit",3),("lateral_move",3),("exfiltrate",3)]
ATT_HARD   = [("scan",1),("escalate",0),("exploit",1),("lateral_move",1),("scan",2),("escalate",1),("exploit",2),("lateral_move",2),("exploit",3),("lateral_move",3),("exfiltrate",3)]

DEF_EASY   = [("monitor",1),("patch",1),("isolate",1)]
DEF_MEDIUM = [("monitor",3),("patch",2),("restore",1),("monitor",2),("patch",3),("monitor",1)]
DEF_HARD   = [("monitor",3),("patch",2),("isolate",3),("restore",1),("block",2),("monitor",1),("patch",3),("isolate",2)]

TASKS = [
    ("easy",   ATT_EASY,   DEF_EASY),
    ("medium", ATT_MEDIUM, DEF_MEDIUM),
    ("hard",   ATT_HARD,   DEF_HARD),
]

print(SEP)
print("  DUAL-ROLE EVALUATION: 6 SCENARIOS (3 tasks x 2 roles)")
print("  Same agent evaluates as ATTACKER then DEFENDER on each task")
print(SEP)
print()

all_scores = {}
for task, att, defs in TASKS:
    print()
    print("[SCENARIO %s - ATTACKER ROLE]" % task.upper())
    s_att = battle(task, "attacker", att, defs)
    all_scores[task+"_att"] = s_att

    print()
    print("[SCENARIO %s - DEFENDER ROLE]" % task.upper())
    s_def = battle(task, "defender", att, defs)
    all_scores[task+"_def"] = s_def

    combined = (s_att + s_def) / 2
    bar_att = int(s_att*20)*"#" + (20-int(s_att*20))*"."
    bar_def = int(s_def*20)*"#" + (20-int(s_def*20))*"."
    print()
    print("  ATT [%s] %.0f%%" % (bar_att, s_att*100))
    print("  DEF [%s] %.0f%%" % (bar_def, s_def*100))
    print("  COMBINED SCORE: %.2f (%.0f%%)" % (combined, combined*100))
    print()
    print(SEP)

overall = sum(all_scores.values()) / len(all_scores)
print()
print("  FINAL DUAL-ROLE RESULTS")
print("  %s" % ("-"*50))
for k,v in all_scores.items():
    bar = int(v*20)*"#" + (20-int(v*20))*"."
    print("  %-12s [%s] %.0f%%" % (k, bar, v*100))
print("  %s" % ("-"*50))
print("  OVERALL DUAL-ROLE SCORE: %.2f (%.0f%%)" % (overall, overall*100))
print()
thresh = 0.4
passed = sum(1 for v in all_scores.values() if v >= thresh)
print("  Scenarios passing threshold (%.0f%%): %d/6" % (thresh*100, passed))
print()
if overall >= 0.5:
    print("  STATUS: EXCELLENT - Agent masters both attack and defense!")
elif overall >= 0.35:
    print("  STATUS: GOOD - Agent shows solid dual-role capability")
else:
    print("  STATUS: LEARNING - More training needed on some scenarios")
print(SEP)
