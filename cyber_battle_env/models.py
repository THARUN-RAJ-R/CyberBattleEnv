"""
CyberBattleEnv — Pydantic models for Action, Observation, State.

Inherits from openenv-core base classes when available,
falls back to plain Pydantic models otherwise so the server
can run without openv-core installed during local development.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ── openenv-core integration ──────────────────────────────────────────────────
try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore

    _OPENENV_CORE = True
except ImportError:

    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: Optional[float] = None

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0

    _OPENENV_CORE = False


# ── Enumerations ──────────────────────────────────────────────────────────────

class AttackerActionType(str, Enum):
    """Actions available to the attacker agent."""
    SCAN          = "scan"           # Reconnaissance — very low noise
    EXPLOIT       = "exploit"        # Attempt to compromise adjacent node
    LATERAL_MOVE  = "lateral_move"   # Traverse already-compromised node
    ESCALATE      = "escalate"       # Privilege escalation on current node
    EXFILTRATE    = "exfiltrate"     # Extract data from Database (node 3)


class DefenderActionType(str, Enum):
    """Actions available to the automated defender."""
    PATCH    = "patch"    # Reduce vulnerability level
    MONITOR  = "monitor"  # Increase detection sensitivity
    ISOLATE  = "isolate"  # Network-isolate a node
    RESTORE  = "restore"  # Remove compromised status
    BLOCK    = "block"    # Block attacker's current position


class AgentRole(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"


# ── Action ────────────────────────────────────────────────────────────────────

class CyberBattleAction(Action):
    """
    Unified action for the attacker agent.

    For all 3 tasks the caller controls the attacker; the defender
    is scripted inside the environment for tasks 2 & 3.

    action_type: one of AttackerActionType values
    target_node: 0=User, 1=WebServer, 2=AppServer, 3=Database
    """

    agent: AgentRole = AgentRole.ATTACKER
    action_type: str = Field(
        ...,
        description="AttackerActionType value: scan | exploit | lateral_move | escalate | exfiltrate",
    )
    target_node: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Target node ID (0-3)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra parameters",
    )


# ── Observation sub-types ─────────────────────────────────────────────────────

class NodeState(BaseModel):
    """Visible state of a single network node."""

    node_id: int
    name: str
    vulnerability_level: float = Field(description="0.0 (hardened) → 1.0 (fully vulnerable). -1 = unknown")
    is_compromised: bool
    detection_risk: float = Field(description="0.0 (safe) → 1.0 (certain detection)")
    patch_level: float = Field(description="0.0 (unpatched) → 1.0 (fully patched)")
    is_isolated: bool
    is_monitored: bool
    is_visible: bool = True


class DefenderAlert(BaseModel):
    """Alert emitted when suspicious activity is detected."""

    turn: int
    node_id: int
    action_type: str
    severity: float  # 0.0–1.0


# ── Observation ───────────────────────────────────────────────────────────────

class CyberBattleObservation(Observation):
    """
    Full network + game state returned after reset() or step().
    `done` and `reward` are inherited from Observation base.
    """

    # Network
    nodes: List[NodeState]
    attacker_position: int

    # Detection
    attacker_detected: bool
    detection_count: int
    alerts: List[DefenderAlert] = Field(default_factory=list)

    # Progress
    compromised_nodes: List[int]

    # Episode context
    task: str
    turn: int
    max_turns: int

    # Action feedback
    last_action_success: bool
    last_action_message: str
    available_attacker_actions: List[str]

    # Defender state (task 3 info)
    defender_last_action: Optional[str] = None
    defender_last_target: Optional[int] = None


# ── State ─────────────────────────────────────────────────────────────────────

class CyberBattleState(State):
    """
    Episode metadata returned by state() endpoint.
    `episode_id` and `step_count` are inherited from State base.
    """

    task: str
    attacker_position: int
    compromised_nodes: List[int]
    detection_count: int
    total_reward: float
    is_game_over: bool
    winner: Optional[str] = None   # "attacker" | "defender" | None
    phase: str = "active"          # "active" | "ended"
