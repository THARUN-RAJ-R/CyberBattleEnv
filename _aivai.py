import sys, logging, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

env = CyberBattleEnvironment()
obs = env.reset(task="hard", seed=42)

ATT = [("scan",1),("exploit",1),("lateral_move",1),("scan",2),("exploit",2),("lateral_move",2),("exploit",3),("lateral_move",3),("exfiltrate",3)]
DEF = [("monitor",3),("patch",2),("isolate",3),("monitor",2),("restore",1),("patch",3),("block",2),("monitor",1)]

print("TURN  SIDE       ACTION               REWARD   MSG")
print("-"*72)
ai = di = 0

for turn in range(1, 15):
    if obs.done:
        break
    at, an = ATT[ai] if ai < len(ATT) else ("scan",1)
    ai += 1
    obs = env.step(CyberBattleAction(action_type=at, target_node=an))
    r = obs.reward
    msg = obs.last_action_message.encode("ascii","replace").decode()[:28]
    print("%-5d %-10s %-20s %+.2f   %s" % (turn, "ATTACKER", at+"("+str(an)+")", r, msg))
    if obs.done:
        break
    dt, dn = DEF[di] if di < len(DEF) else ("monitor",3)
    di += 1
    dobs = env.apply_defender_action(dt, dn)
    dmsg = dobs.last_action_message.encode("ascii","replace").decode()[:28]
    print("      %-10s %-20s  ---    %s" % ("DEFENDER", dt+"("+str(dn)+")", dmsg))
    obs = dobs
    if obs.done:
        break

print("-"*72)
st = env.state
score = min(max(st.total_reward, 0.0), 1.0)
print()
print("WINNER    : " + str(st.winner or "none (time limit)"))
print("DETECTIONS: " + str(st.detection_count) + " / 3 max")
print("SCORE     : %.2f (%.0f%%)" % (score, score*100))
print()
print("FINAL NODE STATES:")
for n in obs.nodes:
    s = "COMPROMISED" if n.is_compromised else "intact"
    iso = " ISOLATED" if n.is_isolated else ""
    mon = " MONITORED" if n.is_monitored else ""
    print("  Node%d %-16s vuln=%.2f patch=%.2f det=%.2f [%s%s%s]" % (
        n.node_id, n.name, n.vulnerability_level, n.patch_level,
        n.detection_risk, s, iso, mon))
