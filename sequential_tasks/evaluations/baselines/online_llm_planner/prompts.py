"""Prompt templates, world-state formatter, and tool schema for the
online-LLM-planner baseline.

The planner is given:
* the natural-language mission summary,
* a textual snapshot of agent + landmark positions and the joint DFA state,
* a short trailing event log,

and is asked to assign one action per agent via a single tool call.
"""

from __future__ import annotations

from typing import List, Sequence

# Action vocabulary the LLM is allowed to issue per agent.
ACTIONS: Sequence[str] = (
    "navigate_to_casualty_A",
    "navigate_to_casualty_B",
    "navigate_to_stretcher",
    "navigate_to_bay_A",
    "navigate_to_bay_B",
    "idle",
)

# Tool schema for structured output.
ASSIGN_ACTIONS_TOOL = {
    "name": "assign_actions",
    "description": (
        "Assign one action to each agent for the next planning interval. "
        "Reasoning is short (≤ 2 sentences) — used for logs only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_0": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 0 (scout)"},
            "agent_1": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 1 (scout)"},
            "agent_2": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 2 (carrier)"},
            "agent_3": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 3 (carrier)"},
            "reasoning": {"type": "string", "description": "Brief reasoning (≤ 2 sentences)"},
        },
        "required": ["agent_0", "agent_1", "agent_2", "agent_3", "reasoning"],
    },
}

SYSTEM_PROMPT = """You are the central planner for a search-and-rescue robot team.

Team: 4 agents — agent_0 and agent_1 are SCOUTS (fast, can detect casualties), agent_2 and agent_3 are CARRIERS (move the stretcher).

World contains 2 casualties (A red, B orange), 1 stretcher, 2 medical bays (Bay A green, Bay B blue).

Mechanics you must respect:
* Scouts discover a casualty by coming within 0.45 of it.
* The stretcher only carries a casualty if BOTH carriers are within 0.40 of the stretcher in the same step. The stretcher then moves with the centroid of the docked carriers.
* To pick up casualty X: both carriers must be docked at the stretcher AND the stretcher must be within 0.50 of casualty X (and X must be FOUND, not LOADED).
* To deliver casualty X: stretcher (with X loaded) must reach within 0.40 of the matching bay (A → Bay A green, B → Bay B blue).
* Only one casualty can be loaded at a time.
* Mission ends when both casualties are DELIVERED.

Coordination tip: To move the stretcher with a casualty, BOTH carriers should issue the same `navigate_to_*` target so they stay docked while the stretcher follows. If carriers issue different targets, the stretcher does not move.

You will be called repeatedly during the episode. Each time, output one action per agent via the assign_actions tool. Respect the mission's ordering / priority constraints (e.g. "rescue red first" must be obeyed).
"""


def format_world_state(
    *,
    summary: str,
    step: int,
    max_steps: int,
    agent_positions: Sequence[Sequence[float]],   # 4 × 2
    agent_velocities: Sequence[Sequence[float]],  # 4 × 2
    agent_roles: Sequence[str],                    # 4 strings: "scout"/"carrier"
    casualty_positions: Sequence[Sequence[float]], # 2 × 2
    casualty_states: Sequence[str],                # 2 strings: unfound/found/loaded/delivered
    stretcher_position: Sequence[float],           # 2
    bay_positions: Sequence[Sequence[float]],      # 2 × 2
    recent_events: Sequence[str],                  # short list, formatted strings
) -> str:
    def _xy(p: Sequence[float]) -> str:
        return f"({float(p[0]):+.2f}, {float(p[1]):+.2f})"

    lines: List[str] = []
    lines.append(f"Mission: {summary}")
    lines.append(f"Step: {step}/{max_steps}")
    lines.append("")
    lines.append("Agents:")
    for i, (pos, vel, role) in enumerate(zip(agent_positions, agent_velocities, agent_roles)):
        lines.append(f"  agent_{i} ({role:>7s})  pos={_xy(pos)}  vel={_xy(vel)}")
    lines.append("")
    lines.append("Landmarks:")
    for letter, idx in (("A", 0), ("B", 1)):
        lines.append(
            f"  Casualty {letter}: pos={_xy(casualty_positions[idx])}  status={casualty_states[idx]}"
        )
    lines.append(f"  Stretcher:   pos={_xy(stretcher_position)}")
    lines.append(f"  Bay A (green): pos={_xy(bay_positions[0])}")
    lines.append(f"  Bay B (blue):  pos={_xy(bay_positions[1])}")
    lines.append("")
    if recent_events:
        lines.append("Recent events:")
        for ev in recent_events[-5:]:
            lines.append(f"  - {ev}")
    else:
        lines.append("Recent events: (none yet)")
    lines.append("")
    lines.append("Plan the next actions for each agent. Use the assign_actions tool.")
    return "\n".join(lines)
