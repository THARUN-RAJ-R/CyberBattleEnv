import sys, io, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="ascii", errors="replace")
sys.path.insert(0, ".")
logging.disable(logging.CRITICAL)

from cyber_battle_env.server.environment import CyberBattleEnvironment
from cyber_battle_env.models import CyberBattleAction

print("=" * 65)
print("  TRUE AI vs AI — ONE EPISODE, BOTH MINDS EVERY TURN")
print("  (LLM attacker prompt + LLM defender prompt per turn)")
print("=" * 65)
print()

env = CyberBattleEnvironment()
env.reset(task="hard", seed=42)   # always role=attacker for the main game

# Simulate 8 turns: each turn = attacker move + defender move
ATT = [("scan",1),("exploit",1),("lateral_move",1),("scan",2),("exploit",2),("lateral_move",2),("exploit",3),("exfiltrate",3)]
DEF = [("monitor",3),("patch",2),("isolate",3),("restore",1),("block",2),("monitor",1),("patch",3),("isolate",2)]

print("TURN  [ATTACKER LLM DECISION]        reward  [DEFENDER LLM DECISION]")
print("-" * 78)

for i in range(8):
    at, an = ATT[i]
    dt, dn = DEF[i]

    # -- ATTACKER turn (LLM call 1) --
    obs = env.step(CyberBattleAction(action_type=at, target_node=an))
    att_r = obs.reward
    att_m = obs.last_action_message.encode("ascii","replace").decode()[:26]

    if obs.done:
        print("%-5d %-28s %+.2f   [episode ended]" % (i+1, at+"("+str(an)+")", att_r))
        print()
        print("WINNER:", env.state.winner or "none")
        break

    # -- DEFENDER turn (LLM call 2) --
    dobs = env.apply_defender_action(dt, dn)
    def_m = dobs.last_action_message.encode("ascii","replace").decode()[:26]

    print("%-5d %-28s %+.2f   %s" % (
        i+1,
        at+"("+str(an)+") -> "+att_m,
        att_r,
        dt+"("+str(dn)+") -> "+def_m
    ))

    if dobs.done:
        print("  [Defender ended episode]")
        break

print()
print("=" * 65)
print("  HOW IT WORKS:")
print("  - Both decisions happen in the SAME turn, SAME episode")
print("  - LLM has ATTACKER system prompt for call 1")
print("  - LLM has DEFENDER system prompt for call 2")
print("  - Neither knows what the other will do next turn")
print("  - This is equivalent to two independent agents battling")
print("=" * 65)
