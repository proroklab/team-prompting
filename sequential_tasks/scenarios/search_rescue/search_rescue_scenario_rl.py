"""BenchMARL training entry point for the Search-and-Rescue scenario.

Subclasses ``SearchRescueScenario`` and adds:

* ``observation`` — returns a dict with pos / vel / sentence_embedding / obs
  keys (the contract BenchMARL's GNN policy expects, mirroring the four_flags
  pipeline).
* ``pre_step`` — refreshes per-agent goal_coords each step (the joint state
  may have changed since the last reset).
* ``extra_render`` — minimal visualisation: walls, agent role colour, comms
  graph, current task summary text.

Used as the scenario class in ``conf/search_rescue/task/search_rescue_rl.yaml``.
"""

from __future__ import annotations

import torch
from vmas.simulator.core import Agent, Line
from vmas.simulator.utils import Color
from vmas import render_interactively

from scenarios.search_rescue.search_rescue_scenario import (
    SearchRescueScenario as BaseSearchRescueScenario,
    ROLE_SCOUT, ROLE_CARRIER,
    N_CASUALTIES, N_BAYS,
)


class SearchRescueScenario(BaseSearchRescueScenario):
    """Adds a BenchMARL-compatible observation API on top of the base scenario."""

    def observation(self, agent: Agent):
        B = self.world.batch_dim
        device = self.world.device
        e = getattr(self, "_cached_event_vec", torch.zeros((B, self.event_dim), device=device))

        own_pos = agent.state.pos
        own_vel = agent.state.vel

        # Flat "obs": role + relative landmark positions + event vec + sub-goal direction.
        obs_buf = torch.empty((B, self._obs_dim - 4), device=device, dtype=torch.float32)
        i = 0
        obs_buf[:, i:i + 2] = agent.role_one_hot;                       i += 2
        obs_buf[:, i:i + 2] = self.casualties[0].state.pos - own_pos;   i += 2
        obs_buf[:, i:i + 2] = self.casualties[1].state.pos - own_pos;   i += 2
        obs_buf[:, i:i + 2] = self.stretcher.state.pos - own_pos;       i += 2
        obs_buf[:, i:i + 2] = self.bays[0].state.pos - own_pos;         i += 2
        obs_buf[:, i:i + 2] = self.bays[1].state.pos - own_pos;         i += 2
        obs_buf[:, i:i + self.event_dim] = e;                            i += self.event_dim
        # Sub-goal direction: this is what symmetry-breaks scouts (scout_0's
        # goal_coords is casualty A, scout_1's is casualty B) and tells carriers
        # what they should be hauling toward. Without this, the shared policy
        # has no per-agent observation difference and all scouts converge on
        # the same casualty (eval video iter 399 confirmed this failure mode).
        obs_buf[:, i:i + 2] = agent.goal_coords - own_pos;              i += 2
        # ``self._obs_dim`` includes own_pos+own_vel; obs_buf above excludes
        # them because BenchMARL's GNN model takes pos/vel separately.

        return {
            "pos": own_pos,
            "vel": own_vel,
            "sentence_embedding": agent.y,
            "obs": obs_buf,
        }

    def pre_step(self):
        for agent in self.world.agents:
            self._set_goal_coords(agent, env_index=None)

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        geoms = []

        # Walls
        for i in range(4):
            is_vertical = (i % 2 == 0)
            length = (
                2 * self.world.y_semidim + 2 * self.agent_radius if is_vertical
                else 2 * self.world.x_semidim
            )
            geom = Line(length=length).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            cx, cy = (
                ( self.world.x_semidim, 0.0) if i == 0 else
                ( 0.0,  self.world.y_semidim + self.agent_radius) if i == 1 else
                (-self.world.x_semidim, 0.0) if i == 2 else
                ( 0.0, -self.world.y_semidim - self.agent_radius)
            )
            xform.set_translation(cx, cy)
            xform.set_rotation(float(torch.pi / 2) if is_vertical else 0.0)
            geom.set_color(*Color.BLACK.value)
            geoms.append(geom)

        # Comms graph (carrier-to-carrier and scout-to-scout edges within radius)
        for i, a1 in enumerate(self.world.agents):
            for j, a2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                d = self.world.get_distance(a1, a2)[env_index]
                if d <= self.comms_radius:
                    line = rendering.Line(a1.state.pos[env_index], a2.state.pos[env_index], width=3)
                    line.set_color(0.4, 0.4, 0.4)
                    geoms.append(line)

        # Print summary text if available
        try:
            sentence = self.language_unit.summary[env_index]
            if sentence:
                text = rendering.TextLine(text=sentence, font_size=6)
                text.label.color = (0, 0, 0, 255)
                xform = rendering.Transform()
                text.add_attr(xform)
                geoms.append(text)
        except Exception:
            pass

        return geoms


if __name__ == "__main__":
    scenario = SearchRescueScenario()
    render_interactively(scenario, control_two_agents=True, display_info=False)
