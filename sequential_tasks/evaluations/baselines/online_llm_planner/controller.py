"""Low-level controller: turn a per-agent landmark choice into a velocity command.

Simple proportional controller — drives the agent toward the target landmark
position, capped at ``v_range``. The scenario uses a velocity controller so
the action vector is interpreted as a target velocity directly.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch


# Map LLM action enum → "landmark name" for landmark-position lookup.
# "idle" is special-cased.
ACTION_TO_LANDMARK = {
    "navigate_to_casualty_A": "casualty_A",
    "navigate_to_casualty_B": "casualty_B",
    "navigate_to_stretcher":  "stretcher",
    "navigate_to_bay_A":      "bay_A",
    "navigate_to_bay_B":      "bay_B",
    "idle":                   None,
}


def landmark_positions(scenario, env_index: int = 0) -> Dict[str, torch.Tensor]:
    """Return a dict landmark_name → 2D position (per-env tensor of shape (2,)).

    For batched envs, the caller should slice with ``env_index``.
    """
    return {
        "casualty_A": scenario.casualties[0].state.pos[env_index],
        "casualty_B": scenario.casualties[1].state.pos[env_index],
        "stretcher":  scenario.stretcher.state.pos[env_index],
        "bay_A":      scenario.bays[0].state.pos[env_index],
        "bay_B":      scenario.bays[1].state.pos[env_index],
    }


def actions_to_velocities(
    *,
    scenario,
    actions_per_agent: Sequence[str],
    env_index: int,
    v_max: float,
    gain: float = 2.0,
) -> torch.Tensor:
    """Convert list of action strings → (n_agents, 2) velocity tensor for one env.

    ``actions_per_agent`` length must equal ``len(scenario.world.agents)``.
    Returns CPU/CUDA tensor matching the device of agent positions.
    """
    device = scenario.world.agents[0].state.pos.device
    n = len(scenario.world.agents)
    out = torch.zeros((n, 2), device=device)

    landmarks = landmark_positions(scenario, env_index=env_index)

    for i, action in enumerate(actions_per_agent):
        landmark_name = ACTION_TO_LANDMARK.get(action)
        if landmark_name is None:
            continue  # idle → zero velocity
        target = landmarks[landmark_name]
        own_pos = scenario.world.agents[i].state.pos[env_index]
        delta = target - own_pos
        dist = torch.linalg.vector_norm(delta)
        if dist < 1e-4:
            continue  # essentially at target
        direction = delta / dist
        speed = min(float(dist) * gain, v_max)
        out[i] = direction * speed

    return out
