"""
CyberBattleEnvironment — Core RL simulation.

Network topology:
  [0: User Workstation] ─── [1: Web Server] ─── [2: App Server] ─── [3: Database]

Tasks:
  easy   → Compromise Web Server (node 1). Passive defender.
  medium → Reach & exfiltrate Database stealthily (≤2 detections). Moderate defender.
  hard   → Full battle. Aggressive defender. Win before getting caught 3 times.
"""
from __future__ import annotations

import logging
import random
import uuid
from typing import Dict, List, Optional, Tuple

from ..models import (
    AttackerActionType,
    CyberBattleAction,
    CyberBattleObservation,
    CyberBattleState,
    DefenderActionType,
    DefenderAlert,
    NodeState,
)

logger = logging.getLogger(__name__)

# ── Network constants ─────────────────────────────────────────────────────────

NODE_NAMES: Dict[int, str] = {
    0: "User Workstation",
    1: "Web Server",
    2: "App Server",
    3: "Database",
}

TOPOLOGY: Dict[int, List[int]] = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2],
}

# (vulnerability, patch_level, detection_risk)
NODE_DEFAULTS: Dict[int, Tuple[float, float, float]] = {
    0: (0.0, 1.0, 0.0),   # User — already owned, not attackable
    1: (0.70, 0.20, 0.30), # Web Server — high vuln
    2: (0.50, 0.40, 0.50), # App Server — moderate
    3: (0.30, 0.60, 0.70), # Database — hardened + monitored
}

MAX_TURNS:       Dict[str, int] = {"easy": 15, "medium": 20, "hard": 25}
MAX_DETECTIONS:  Dict[str, int] = {"easy": 99, "medium": 2,  "hard": 3}

# Reward table per node for successful exploit
EXPLOIT_REWARDS: Dict[int, float] = {0: 0.02, 1: 0.35, 2: 0.25, 3: 0.20}

# Lateral-move progress rewards
MOVE_REWARDS: Dict[int, float] = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.15}


# ── Internal node model ───────────────────────────────────────────────────────

class _Node:
    """Mutable node state, kept inside the environment only."""

    def __init__(self, node_id: int, vuln: float, patch: float, det_risk: float) -> None:
        self.node_id = node_id
        self.name = NODE_NAMES[node_id]
        self.vulnerability_level = vuln
        self.patch_level = patch
        self.detection_risk = det_risk
        self.is_compromised = node_id == 0  # attacker starts with node-0
        self.is_isolated = False
        self.is_monitored = False
        self.scanned = False

    def to_model(self) -> NodeState:
        return NodeState(
            node_id=self.node_id,
            name=self.name,
            vulnerability_level=round(self.vulnerability_level, 3),
            is_compromised=self.is_compromised,
            detection_risk=round(self.detection_risk, 3),
            patch_level=round(self.patch_level, 3),
            is_isolated=self.is_isolated,
            is_monitored=self.is_monitored,
            is_visible=True,
        )


# ── Environment ───────────────────────────────────────────────────────────────

class CyberBattleEnvironment:
    """
    OpenEnv-compatible RL environment for cybersecurity simulation.
    Thread-safe per session; set SUPPORTS_CONCURRENT_SESSIONS = True.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ── init / reset ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._rng = random.Random()
        self._nodes: Dict[int, _Node] = {}
        self._attacker_pos = 0
        self._task = "easy"
        self._role = "attacker"   # "attacker" | "defender"
        self._turn = 0
        self._detection_count = 0
        self._alerts: List[DefenderAlert] = []
        self._episode_id: Optional[str] = None
        self._total_reward = 0.0
        self._done = False
        self._winner: Optional[str] = None
        self._compromised: List[int] = [0]
        self._scripted_att_idx = 0   # index into scripted attacker playbook

    def reset(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        role: str = "attacker",
        **kwargs,
    ) -> CyberBattleObservation:
        """Start a new episode.
        task ∈ {"easy", "medium", "hard"}
        role ∈ {"attacker", "defender"}
          attacker — agent controls attacker, scripted defender reacts after each step.
          defender — agent controls defender, scripted attacker acts before each step.
        """
        if task not in MAX_TURNS:
            raise ValueError(f"Unknown task '{task}'. Choose: easy | medium | hard")
        if role not in ("attacker", "defender"):
            raise ValueError(f"Unknown role '{role}'. Choose: attacker | defender")

        self._rng = random.Random(seed)
        self._task = task
        self._role = role
        self._turn = 0
        self._detection_count = 0
        self._alerts = []
        self._done = False
        self._winner = None
        self._total_reward = 0.0
        self._defender_reward = 0.0
        self._attacker_pos = 0
        self._compromised = [0]
        self._episode_id = episode_id or str(uuid.uuid4())
        self._scripted_att_idx = 0

        self._nodes = {
            nid: _Node(nid, v, p, d)
            for nid, (v, p, d) in NODE_DEFAULTS.items()
        }

        logger.info("[RESET] episode=%s task=%s role=%s", self._episode_id, task, role)
        role_msg = (
            "DEFENDER mode: protect the network from the scripted attacker."
            if role == "defender" else
            f"ATTACKER mode: Task={task}. Start at node 0 (User Workstation)."
        )
        return self._build_obs(done=False, reward=0.0, success=True, msg=role_msg)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: CyberBattleAction, **kwargs) -> CyberBattleObservation:
        """Process one step. Behaviour depends on role:
          attacker role -> action is an attacker action; scripted defender reacts after.
          defender role -> action is a defender action; scripted attacker acts first.
        """
        if self._done:
            return self._build_obs(
                done=True, reward=0.0, success=False,
                msg="Episode ended. Call reset() to start a new episode.",
            )
        if self._role == "defender":
            return self._step_as_defender(action)

        self._turn += 1
        self._alerts = []

        obs, step_reward = self._dispatch_attacker(action)

        # Scripted defender reacts after the attacker (tasks 2 & 3)
        def_action, def_target = None, None
        if not self._done and self._task in ("medium", "hard"):
            def_action, def_target = self._scripted_defender()

        # Time-limit check
        if not self._done and self._turn >= MAX_TURNS[self._task]:
            self._done = True
            step_reward = max(-0.1, step_reward - 0.1)
            obs = self._build_obs(
                done=True, reward=step_reward, success=False,
                msg="Time limit reached.",
                def_action=def_action, def_target=def_target,
            )
        else:
            obs.defender_last_action = def_action
            obs.defender_last_target = def_target

        self._total_reward = max(0.00, min(1.00, self._total_reward + step_reward))
        logger.info(
            "[STEP] turn=%d action=%s@%d reward=%.3f done=%s",
            self._turn, action.action_type, action.target_node,
            step_reward, self._done,
        )
        return obs

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> CyberBattleState:
        return CyberBattleState(
            episode_id=self._episode_id,
            step_count=self._turn,
            task=self._task,
            attacker_position=self._attacker_pos,
            compromised_nodes=list(self._compromised),
            detection_count=self._detection_count,
            total_reward=round(self._total_reward, 4),
            is_game_over=self._done,
            winner=self._winner,
            phase="ended" if self._done else "active",
        )

    # ── attacker action dispatcher ────────────────────────────────────────────

    def _dispatch_attacker(
        self, action: CyberBattleAction
    ) -> Tuple[CyberBattleObservation, float]:
        atype = action.action_type
        target = action.target_node

        # Validate action type
        valid_types = [a.value for a in AttackerActionType]
        if atype not in valid_types:
            return self._fail(f"Unknown action '{atype}'. Valid: {valid_types}", -0.05)

        if atype == AttackerActionType.SCAN:
            return self._do_scan(target)
        if atype == AttackerActionType.EXPLOIT:
            return self._do_exploit(target)
        if atype == AttackerActionType.LATERAL_MOVE:
            return self._do_lateral_move(target)
        if atype == AttackerActionType.ESCALATE:
            return self._do_escalate(target)
        if atype == AttackerActionType.EXFILTRATE:
            return self._do_exfiltrate(target)
        return self._fail(f"Unhandled action: {atype}", -0.05)

    # ── individual actions ────────────────────────────────────────────────────

    def _do_scan(self, target: int) -> Tuple[CyberBattleObservation, float]:
        node = self._nodes.get(target)
        if node is None:
            return self._fail(f"Node {target} does not exist.", -0.05)

        if target not in TOPOLOGY.get(self._attacker_pos, []) and target != self._attacker_pos:
            return self._fail(f"Node {target} not reachable from position {self._attacker_pos}.", -0.02)
            
        if getattr(node, "scanned", False):
            return self._fail(f"Node {target} already scanned. Redundant action.", -0.01)

        node.scanned = True
        detected = self._detect(node, stealth=0.8)
        reward = 0.06 if not node.is_compromised else 0.03
        if detected:
            reward = max(0.0, reward - 0.11)

        msg = (f"Scanned {node.name}: vuln={node.vulnerability_level:.2f}, "
               f"patch={node.patch_level:.2f}, det={node.detection_risk:.2f}")
        if detected:
            msg += " [DETECTED]"
        return self._build_obs(done=self._done, reward=reward, success=True, msg=msg), reward

    def _do_exploit(self, target: int) -> Tuple[CyberBattleObservation, float]:
        node = self._nodes.get(target)
        if node is None:
            return self._fail(f"Node {target} does not exist.", -0.05)
        if node.is_compromised:
            return self._fail(f"{node.name} already compromised.", -0.02)
        if node.is_isolated:
            return self._fail(f"{node.name} is isolated — cannot exploit.", -0.03)

        adj = TOPOLOGY.get(self._attacker_pos, [])
        if target not in adj:
            return self._fail(f"{node.name} not adjacent to {NODE_NAMES[self._attacker_pos]}.", -0.02)

        p_success = node.vulnerability_level * (1.0 - node.patch_level) * 0.85
        p_success = min(max(p_success, 0.05), 0.95)
        success = self._rng.random() < p_success
        detected = self._detect(node, stealth=0.4)

        reward = 0.0
        if success:
            node.is_compromised = True
            if target not in self._compromised:
                self._compromised.append(target)
            reward = EXPLOIT_REWARDS.get(target, 0.2)
            msg = f"✓ Exploit succeeded on {node.name} (p={p_success:.2f})"
        else:
            reward = -0.04
            msg = f"✗ Exploit failed on {node.name} (p={p_success:.2f})"

        if detected:
            reward -= 0.22
            msg += " [DETECTED]"
        elif success:
            reward += 0.05
            msg += " [STEALTH BONUS]"

        # Task 1 win: web server compromised
        if self._task == "easy" and 1 in self._compromised:
            reward += 0.5
            self._done = True
            self._winner = "attacker"
            msg += " 🏆 Task 1 complete — Web Server compromised!"

        final_reward = max(-0.2, reward)
        return self._build_obs(done=self._done, reward=final_reward, success=success, msg=msg), final_reward

    def _do_lateral_move(self, target: int) -> Tuple[CyberBattleObservation, float]:
        if target == self._attacker_pos:
            return self._fail("Already at this node.", -0.01)

        node = self._nodes.get(target)
        if node is None:
            return self._fail(f"Node {target} does not exist.", -0.05)
        if target not in TOPOLOGY.get(self._attacker_pos, []):
            return self._fail(f"{node.name} not adjacent.", -0.02)
        if not node.is_compromised:
            return self._fail(f"{node.name} not compromised. Use exploit first.", -0.02)
        if node.is_isolated:
            return self._fail(f"{node.name} isolated — cannot enter.", -0.03)

        prev = self._attacker_pos
        self._attacker_pos = target
        detected = self._detect(node, stealth=0.6)
        reward = MOVE_REWARDS.get(target, 0.05)
        if detected:
            reward = max(-0.10, reward - 0.10)

        msg = f"Moved {NODE_NAMES[prev]} → {node.name}"
        if detected:
            msg += " [DETECTED]"
        else:
            reward += 0.05
            msg += " [GHOST PROTOCOL BONUS]"
        return self._build_obs(done=self._done, reward=reward, success=True, msg=msg), reward

    def _do_escalate(self, target: int) -> Tuple[CyberBattleObservation, float]:
        node = self._nodes.get(target)
        if node is None or not node.is_compromised:
            return self._fail(f"Cannot escalate: {NODE_NAMES.get(target,'?')} not compromised.", -0.03)
        if target not in (self._attacker_pos, *TOPOLOGY.get(self._attacker_pos, [])):
            return self._fail("Cannot escalate remote node.", -0.02)

        node.detection_risk = max(0.0, node.detection_risk - 0.15)
        detected = self._detect(node, stealth=0.5)
        reward = 0.07 if not detected else 0.02

        msg = f"Privilege escalation on {node.name}. Detection risk reduced."
        if detected:
            msg += " [DETECTED]"
        return self._build_obs(done=self._done, reward=reward, success=True, msg=msg), reward

    def _do_exfiltrate(self, target: int) -> Tuple[CyberBattleObservation, float]:
        if target != 3:
            return self._fail("Exfiltration only valid on Database (node 3).", -0.05)

        db = self._nodes[3]
        if not db.is_compromised:
            return self._fail("Database not compromised. Cannot exfiltrate.", -0.05)
        if self._attacker_pos not in (3, *TOPOLOGY.get(3, [])):
            return self._fail("Too far from Database to exfiltrate.", -0.03)

        detected = self._detect(db, stealth=0.15)

        if detected and self._task == "medium":
            self._done = True
            self._winner = "defender"
            reward = -0.3
            msg = "Exfiltration detected! Caught by security team. [DEFENDER WINS]"
        else:
            reward = 0.7
            self._done = True
            self._winner = "attacker"
            if detected:
                reward -= 0.15
            msg = f"💀 Data exfiltrated! Task '{self._task}' complete."
            if detected:
                msg += " (Partial detection during extraction)"

        return self._build_obs(done=self._done, reward=reward, success=(self._winner == "attacker"), msg=msg), reward

    # ── scripted defender ─────────────────────────────────────────────────────

    def _scripted_defender(self) -> Tuple[Optional[str], Optional[int]]:
        if self._task == "medium":
            return self._defender_moderate()
        return self._defender_aggressive()

    def apply_defender_action(self, action_type: str, target_node: int) -> "CyberBattleObservation":
        """
        Apply a DEFENDER action chosen by the LLM (TRUE AI vs AI mode).
        Called by POST /defender_step. Available: patch, monitor, isolate, restore, block.
        """
        node = self._nodes.get(target_node)
        msg  = "Defender: "
        gained = 0.0

        if action_type == "patch":
            if node:
                if node.patch_level >= 1.0:
                    msg += f"Wasteful Time! {node.name} is entirely updated."
                    gained = -0.06
                else:
                    node.patch_level         = min(1.0, node.patch_level + 0.25)
                    node.vulnerability_level = max(0.0, node.vulnerability_level - 0.20)
                    msg += "Patched " + node.name
                    gained = 0.12
            else:
                msg += "patch failed — bad node"
                gained = -0.08
        elif action_type == "monitor":
            if node:
                node.is_monitored   = True
                node.detection_risk = min(1.0, node.detection_risk + 0.25)
                msg += "Monitoring " + node.name
                gained = 0.07
            else:
                msg += "monitor failed — bad node"
                gained = -0.08
        elif action_type == "isolate":
            if node and target_node != 0:
                node.is_isolated = True
                if not getattr(node, "is_compromised", False):
                    msg += f"Severe Business Disruption! Isolated clean {node.name}"
                    gained = -0.15
                else:
                    msg += "Isolated " + node.name
                    gained = 0.18
            else:
                msg += "isolate failed — invalid target"
                gained = -0.08
        elif action_type == "restore":
            if node and target_node != 0:
                if getattr(node, "is_compromised", False) == False:
                    msg += f"Wasteful response. {node.name} is already clean."
                    gained = -0.08
                else:
                    node.is_compromised = False
                    if target_node in self._compromised:
                        self._compromised.remove(target_node)
                    msg += "Restored " + node.name
                    gained = 0.34
            else:
                msg += "restore failed — invalid"
                gained = -0.08
        elif action_type == "block":
            if node:
                node.detection_risk = 1.0
                node.is_monitored   = True
                if target_node == 3:
                    msg += "Critical Database Guard! Block active on Vault! "
                    gained = 0.25
                else:
                    msg += "Blocked " + node.name + " (det=1.0)"
                    gained = 0.11
            else:
                msg += "block failed — bad node"
                gained = -0.08
        else:
            msg += "unknown action '" + action_type + "'"
            gained = -0.05

        if not hasattr(self, "_defender_reward"):
            self._defender_reward = 0.0
        self._defender_reward = max(0.00, min(1.00, self._defender_reward + gained))

        return self._build_obs(done=self._done, reward=0.0, success=True, msg=msg)




    # __ Defender role: step_as_defender __
    _ATT_PLAYBOOKS = {
        "easy":   [("scan", 1), ("exploit", 1)],
        "medium": [("scan", 1), ("exploit", 1), ("lateral_move", 1),
                   ("scan", 2), ("exploit", 2), ("lateral_move", 2),
                   ("scan", 3), ("exploit", 3), ("lateral_move", 3), ("exfiltrate", 3)],
        "hard":   [("scan", 1), ("escalate", 0), ("exploit", 1), ("lateral_move", 1),
                   ("scan", 2), ("escalate", 1), ("exploit", 2), ("lateral_move", 2),
                   ("scan", 3), ("exploit", 3), ("lateral_move", 3),
                   ("escalate", 3), ("exfiltrate", 3)],
    }

    def _scripted_attacker_step(self):
        """Execute next scripted attacker move. Returns (action_type, result_msg)."""
        playbook = self._ATT_PLAYBOOKS.get(self._task, self._ATT_PLAYBOOKS["easy"])
        if self._scripted_att_idx >= len(playbook):
            return "scan", "Scripted attacker: all moves exhausted."
        atype, target = playbook[self._scripted_att_idx]
        self._scripted_att_idx += 1
        action = CyberBattleAction(action_type=atype, target_node=target)
        obs, _ = self._dispatch_attacker(action)
        return atype, obs.last_action_message

    def _step_as_defender(self, action):
        """Defender-role step: scripted attacker moves first, then agent defends."""
        self._turn += 1
        self._alerts = []
        comp_before = len(self._compromised)
        det_before  = self._detection_count
        att_atype, att_msg = self._scripted_attacker_step()
        if self._done:
            step_reward = -0.4
            self._total_reward += step_reward
            return self._build_obs(done=True, reward=step_reward, success=False,
                msg="Attacker exfiltrated! [ATTACKER WINS]")
        comp_after = len(self._compromised)
        det_after  = self._detection_count
        def_obs = self.apply_defender_action(action.action_type, action.target_node)
        step_reward  = 0.0
        step_reward += (det_after - det_before) * 0.15
        step_reward -= (comp_after - comp_before) * 0.20
        step_reward += 0.05
        if self._done and self._winner == "defender":
            step_reward += 0.5
        if not self._done and self._turn >= MAX_TURNS[self._task]:
            self._done   = True
            self._winner = "defender"
            step_reward  += 0.2
        step_reward = round(max(-0.5, min(0.7, step_reward)), 4)
        self._total_reward += step_reward
        msg = "[ATT] " + att_msg[:38] + " | [DEF] " + def_obs.last_action_message[:38]
        return self._build_obs(done=self._done, reward=step_reward, success=True, msg=msg)
    def _defender_moderate(self) -> Tuple[Optional[str], Optional[int]]:
        """Patch the most vulnerable uncompromised node every 3 turns."""
        if self._turn % 3 != 0:
            return None, None
        candidates = [n for n in self._nodes.values() if not n.is_compromised and n.vulnerability_level > 0.2]
        if not candidates:
            return None, None
        target = max(candidates, key=lambda n: n.vulnerability_level)
        target.patch_level = min(1.0, target.patch_level + 0.20)
        target.vulnerability_level = max(0.0, target.vulnerability_level - 0.15)
        return DefenderActionType.PATCH, target.node_id

    def _defender_aggressive(self) -> Tuple[Optional[str], Optional[int]]:
        """Aggressive defender: isolate DB, restore nodes, monitor, patch."""
        # 1. Isolate DB if attacker is at App Server or DB
        if self._attacker_pos >= 2 and not self._nodes[3].is_isolated:
            self._nodes[3].is_isolated = True
            return DefenderActionType.ISOLATE, 3

        # 2. Un-isolate DB if attacker retreated
        if self._attacker_pos < 2 and self._nodes[3].is_isolated:
            self._nodes[3].is_isolated = False

        # 3. Restore a compromised node (not attacker's current)
        for nid in reversed(list(self._compromised)):
            if nid == 0 or nid == self._attacker_pos:
                continue
            self._nodes[nid].is_compromised = False
            self._compromised.remove(nid)
            return DefenderActionType.RESTORE, nid

        # 4. Monitor attacker's position every other turn
        if self._turn % 2 == 0:
            n = self._nodes.get(self._attacker_pos)
            if n:
                n.is_monitored = True
                n.detection_risk = min(1.0, n.detection_risk + 0.20)
            return DefenderActionType.MONITOR, self._attacker_pos

        # 5. Patch most vulnerable uncompromised node
        candidates = [n for n in self._nodes.values() if not n.is_compromised and n.vulnerability_level > 0.1]
        if candidates:
            t = max(candidates, key=lambda n: n.vulnerability_level)
            t.patch_level = min(1.0, t.patch_level + 0.25)
            t.vulnerability_level = max(0.0, t.vulnerability_level - 0.20)
            return DefenderActionType.PATCH, t.node_id

        return None, None

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect(self, node: _Node, stealth: float) -> bool:
        """Return True if attacker is detected. Updates detection counter."""
        risk = node.detection_risk * (1.0 - stealth)
        if node.is_monitored:
            risk = min(1.0, risk + 0.30)
        detected = self._rng.random() < risk
        if detected:
            self._detection_count += 1
            self._alerts.append(DefenderAlert(
                turn=self._turn,
                node_id=node.node_id,
                action_type="attacker_action",
                severity=round(risk, 3),
            ))
            limit = MAX_DETECTIONS.get(self._task, 99)
            if self._detection_count >= limit and not self._done:
                self._done = True
                self._winner = "defender"
        return detected

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fail(self, msg: str, reward: float) -> Tuple[CyberBattleObservation, float]:
        return self._build_obs(done=self._done, reward=reward, success=False, msg=msg), reward

    def _build_obs(
        self,
        done: bool,
        reward: float,
        success: bool,
        msg: str,
        def_action: Optional[str] = None,
        def_target: Optional[int] = None,
    ) -> CyberBattleObservation:
        """Construct a full CyberBattleObservation."""
        node_models = [self._nodes[i].to_model() for i in sorted(self._nodes)]
        available = self._available_actions()
        return CyberBattleObservation(
            done=done,
            reward=reward,
            nodes=node_models,
            attacker_position=self._attacker_pos,
            attacker_detected=bool(self._alerts),
            detection_count=self._detection_count,
            alerts=self._alerts[-3:],
            compromised_nodes=list(self._compromised),
            task=self._task,
            turn=self._turn,
            max_turns=MAX_TURNS.get(self._task, 20),
            last_action_success=success,
            last_action_message=msg,
            available_attacker_actions=available,
            defender_last_action=def_action,
            defender_last_target=def_target,
        )

    def _available_actions(self) -> List[str]:
        """Return the list of valid action strings for the current state."""
        actions: List[str] = []
        adj = TOPOLOGY.get(self._attacker_pos, [])

        for nid in adj:
            actions.append(f"scan({nid})")

        for nid in adj:
            n = self._nodes[nid]
            if not n.is_compromised and not n.is_isolated:
                actions.append(f"exploit({nid})")

        for nid in adj:
            n = self._nodes[nid]
            if n.is_compromised and not n.is_isolated:
                actions.append(f"lateral_move({nid})")

        # Escalate on current node
        cur = self._nodes[self._attacker_pos]
        if cur.is_compromised:
            actions.append(f"escalate({self._attacker_pos})")

        # Exfiltrate from DB
        if self._attacker_pos == 3 or 3 in adj:
            if self._nodes[3].is_compromised:
                actions.append("exfiltrate(3)")

        return actions
