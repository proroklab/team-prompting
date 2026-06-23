"""Search-and-Rescue VMAS scenario (v0.1 — scaffold).

Heterogeneous team (scouts + carriers), two casualties, one stretcher, two
bays. Cooperative transport requires ≥2 carriers docked simultaneously.

This file pairs with:
* sequence_models/search_rescue/random_automaton_walk_search_rescue.py
* scenarios/search_rescue/language.py
* scenarios/search_rescue/load_config.py

TODOs for v0.2 (left as comments inline):
* Goal-coord lookup currently uses simple heuristics; refine once the RNN
  trains and we see what subtask embeddings the LanguageUnit produces.
* Reward weights (``r_*`` in load_config.py) need tuning.
* Stretcher carry physics is a kinematic abstraction (centroid of docked
  carriers) — replace with proper force-coupled dynamics if hardware
  fidelity becomes an issue (note: per user, this task is sim-only).
"""

from __future__ import annotations

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

from scenarios.search_rescue.load_config import load_scenario_config
from scenarios.search_rescue.language import (
    LanguageUnit, load_sequence_model,
    JOINTSTATE_TO_FLAT,
)
from sequence_models.search_rescue.random_automaton_walk_search_rescue import (
    EVENT_DIM,
    UNFOUND, FOUND, LOADED, DELIVERED,
    CASUALTY_A_FOUND, CASUALTY_B_FOUND,
    STRETCHER_LOADED_A, STRETCHER_LOADED_B,
    DELIVERED_A, DELIVERED_B,
    CARRIERS_SYNCED,
    SUB_STATE_INDEX,
)

ROLE_SCOUT, ROLE_CARRIER = 0, 1
N_CASUALTIES = 2
N_BAYS = 2


class SearchRescueScenario(BaseScenario):
    # ================================================================== #
    #  World construction
    # ================================================================== #

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)

        world = World(
            batch_dim, device,
            x_semidim=self.x_semidim, y_semidim=self.y_semidim,
            dim_c=self.comm_dim, collision_force=500, substeps=5,
            linear_friction=self.linear_friction, drag=0,
        )

        self._prepare_obs_layout()
        self._add_casualties(world)
        self._add_stretcher(world)
        self._add_bays(world)
        self._init_language_unit(world)
        self._add_agents(world)

        return world

    # ------------------------------------------------------------------ #

    def _add_casualties(self, world: World) -> None:
        # Casualty A: red-tinted, Casualty B: orange-tinted.
        self.casualty_colors = [(1.0, 0.2, 0.2), (1.0, 0.6, 0.0)]
        self.casualties: list[Landmark] = []
        for i in range(N_CASUALTIES):
            c = Landmark(
                name=f"casualty_{i}",
                collide=False,
                movable=True,  # follows the stretcher when loaded
                shape=Sphere(radius=self.casualty_radius),
                color=self.casualty_colors[i],
            )
            world.add_landmark(c)
            self.casualties.append(c)

    def _add_stretcher(self, world: World) -> None:
        self.stretcher = Landmark(
            name="stretcher",
            collide=False,
            movable=True,
            shape=Box(length=0.6, width=0.3),
            color=(0.6, 0.6, 0.6),
        )
        world.add_landmark(self.stretcher)
        # Per-env carry state
        B = world.batch_dim
        self.stretcher_carrying = torch.full((B,), -1, dtype=torch.long, device=world.device)  # -1, 0, 1

    def _add_bays(self, world: World) -> None:
        # Bay A green, Bay B blue. Position is fixed at opposite corners.
        self.bay_colors = [Color.GREEN, Color.BLUE]
        self.bays: list[Landmark] = []
        for i in range(N_BAYS):
            b = Landmark(
                name=f"bay_{i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.bay_radius),
                color=self.bay_colors[i],
            )
            world.add_landmark(b)
            self.bays.append(b)

    def _init_language_unit(self, world: World) -> None:
        load_sequence_model(
            model_path=self.sequence_model_path,
            embedding_size=self.embedding_size,
            event_size=self.event_dim,
            state_size=self.state_dim,
            device=world.device,
        )
        self.language_unit = LanguageUnit(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
        # Draw the world's semidim boundary lines so the arena edges are
        # visible in the render. (My earlier comment claimed this controlled
        # camera tracking — it doesn't; that's viewer_zoom in load_config.)
        self.visualize_semidims = True
        if self.load_dataset:
            try:
                self.language_unit.load_sequence_data(json_path=self.data_json_path, device=world.device)
            except FileNotFoundError:
                # Augmented dataset doesn't exist yet — fall back to a no-op.
                pass

    def _add_agents(self, world: World) -> None:
        B = world.batch_dim
        # Scouts first, then carriers — keeps role indexing simple
        for i in range(self.n_agents):
            is_scout = i < self.n_scouts
            role = ROLE_SCOUT if is_scout else ROLE_CARRIER

            # All agents share the BenchMARL group name "agents" via the
            # generic name prefix and identical action-space bounds. Role
            # distinction lives in role_one_hot (visible to the policy
            # through the observation dict). Per-role speed differences,
            # if desired, can be applied as a scenario-side multiplier on
            # the agent's commanded velocity rather than via VMAS bounds.
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                v_range=self.agent_v_range,
                f_range=self.agent_f_range,
                u_range=self.agent_u_range,
                color=(0.0, 0.8, 0.4) if is_scout else (0.2, 0.2, 0.9),
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, ctrl_params=(2.0, 6.0, 0.002), pid_form="standard"
                )

            agent.role = role  # 0 scout, 1 carrier
            # role_index: position within the role (0..n_scouts-1 for scouts,
            # 0..n_carriers-1 for carriers). Used by _set_goal_coords to
            # symmetry-break agents within the same role — without this,
            # shared-policy MAPPO collapses to "all scouts go to the same
            # casualty" because they have identical features and parameters.
            agent.role_index = i if is_scout else (i - self.n_scouts)
            agent.role_one_hot = torch.zeros((B, 2), device=world.device)
            agent.role_one_hot[:, role] = 1.0

            agent.h = torch.zeros((B, self.embedding_size), dtype=torch.float32, device=world.device)
            agent.y = torch.zeros((B, self.embedding_size), dtype=torch.float32, device=world.device)
            agent.e = torch.zeros((B, self.event_dim), dtype=torch.float32, device=world.device)
            agent.goal_coords = torch.zeros((B, 2), dtype=torch.float32, device=world.device)

            world.add_agent(agent)

    # ================================================================== #
    #  Observation layout
    # ================================================================== #

    def _prepare_obs_layout(self) -> None:
        """Flat per-agent observation:
            [own_pos(2) | own_vel(2) | role(2) |
             rel_to_casualty_0(2) | rel_to_casualty_1(2) |
             rel_to_stretcher(2) |
             rel_to_bay_0(2) | rel_to_bay_1(2) |
             event_vec(EVENT_DIM) |
             rel_to_goal(2)]    ← per-agent sub-goal direction; symmetry-breaks
                                  scouts (each scout's goal is its assigned
                                  casualty) and gives carriers a strong sub-task
                                  signal once a casualty is FOUND/LOADED.
        """
        i = 0
        self._sl_pos = slice(i, i + 2); i += 2
        self._sl_vel = slice(i, i + 2); i += 2
        self._sl_role = slice(i, i + 2); i += 2
        self._sl_cas0 = slice(i, i + 2); i += 2
        self._sl_cas1 = slice(i, i + 2); i += 2
        self._sl_stretcher = slice(i, i + 2); i += 2
        self._sl_bay0 = slice(i, i + 2); i += 2
        self._sl_bay1 = slice(i, i + 2); i += 2
        self._sl_event = slice(i, i + self.event_dim); i += self.event_dim
        self._sl_goal = slice(i, i + 2); i += 2
        self._obs_dim = i

    # ================================================================== #
    #  Reset
    # ================================================================== #

    def reset_world_at(self, env_index: int | None = None) -> None:
        # Language unit first — joint state determines initial positions / rewards
        if env_index is None:
            self.language_unit.reset_all()
        else:
            self.language_unit.reset_env(env_index)
        self.language_unit.sample_sequence_dataset(env_index)

        self._reset_landmarks(env_index)
        self._reset_agents(env_index)

        if env_index is None:
            self.stretcher_carrying.fill_(-1)
        else:
            self.stretcher_carrying[env_index] = -1

    def _reset_landmarks(self, env_index: int | None) -> None:
        B = self.world.batch_dim
        device = self.world.device

        # Bays: fixed at opposite corners (A=top-left, B=bottom-right)
        bay_positions = torch.tensor(
            [
                [-self.x_semidim + self.bay_radius * 1.5,  self.y_semidim - self.bay_radius * 1.5],
                [ self.x_semidim - self.bay_radius * 1.5, -self.y_semidim + self.bay_radius * 1.5],
            ],
            dtype=torch.float32, device=device,
        )
        for i, bay in enumerate(self.bays):
            pos = bay_positions[i].unsqueeze(0).expand(B, -1) if env_index is None else bay_positions[i]
            bay.set_pos(pos.clone(), batch_index=env_index)

        # Casualties + stretcher: random interior, separated.
        margin = max(self.casualty_radius, self.stretcher_radius) * 1.5
        x_lo, x_hi = -self.x_semidim + margin, self.x_semidim - margin
        y_lo, y_hi = -self.y_semidim + margin, self.y_semidim - margin

        def _rand_pos(batch_size: int) -> torch.Tensor:
            return torch.stack([
                torch.empty(batch_size, device=device).uniform_(x_lo, x_hi),
                torch.empty(batch_size, device=device).uniform_(y_lo, y_hi),
            ], dim=-1)

        bs = B if env_index is None else 1
        # Crude rejection-sample loop for separation.
        max_tries = 8
        cas_pos: list[torch.Tensor] = []
        for _ in range(N_CASUALTIES):
            for _t in range(max_tries):
                p = _rand_pos(bs)
                if not cas_pos:
                    cas_pos.append(p); break
                # Check distance to each existing casualty
                dists = torch.stack([torch.linalg.vector_norm(p - q, dim=-1) for q in cas_pos], dim=0)
                if (dists.min(dim=0).values >= self.min_casualty_separation).all():
                    cas_pos.append(p); break
            else:
                cas_pos.append(p)  # accept last sample

        for i, cas in enumerate(self.casualties):
            cas.set_pos(cas_pos[i] if env_index is None else cas_pos[i].squeeze(0), batch_index=env_index)

        stretcher_pos = _rand_pos(bs)
        self.stretcher.set_pos(
            stretcher_pos if env_index is None else stretcher_pos.squeeze(0),
            batch_index=env_index,
        )

    def _reset_agents(self, env_index: int | None) -> None:
        B = self.world.batch_dim
        device = self.world.device
        x_lo = -self.x_semidim + self.agent_radius
        x_hi =  self.x_semidim - self.agent_radius
        y_lo = -self.y_semidim + self.agent_radius
        y_hi =  self.y_semidim - self.agent_radius
        bs = B if env_index is None else 1

        for agent in self.world.agents:
            pos = torch.stack([
                torch.empty(bs, device=device).uniform_(x_lo, x_hi),
                torch.empty(bs, device=device).uniform_(y_lo, y_hi),
            ], dim=-1)
            agent.set_pos(pos if env_index is None else pos.squeeze(0), batch_index=env_index)

            if env_index is None:
                agent.y = self.language_unit.task_embeddings.clone()
                agent.h = self.language_unit.subtask_embeddings.clone()
                agent.e.zero_()
            else:
                agent.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
                agent.h[env_index] = self.language_unit.subtask_embeddings[env_index].clone()
                agent.e[env_index].zero_()

            self._set_goal_coords(agent, env_index)

    # ================================================================== #
    #  Goal-coord lookup (drives distance shaping)
    # ================================================================== #

    def _set_goal_coords(self, agent: Agent, env_index: int | None = None) -> None:
        """Pick a 2D goal target for ``agent`` based on joint state + role.

        Heuristics (TODO refine):
          * Scout: nearest unfound casualty; if both found, idle near team centroid.
          * Carrier (stretcher empty, some casualty FOUND): go to stretcher.
          * Carrier (stretcher loaded with X): go to bay assigned to X.
          * Carrier (all delivered): idle.
        """
        device = self.world.device
        B = self.world.batch_dim
        idxs = torch.arange(B, device=device) if env_index is None else torch.tensor([env_index], device=device)

        sa = self.language_unit.state_A[idxs]   # 0..3
        sb = self.language_unit.state_B[idxs]
        cas0_pos = self.casualties[0].state.pos[idxs]
        cas1_pos = self.casualties[1].state.pos[idxs]
        stretch_pos = self.stretcher.state.pos[idxs]
        bay0_pos = self.bays[0].state.pos[idxs]
        bay1_pos = self.bays[1].state.pos[idxs]

        if agent.role == ROLE_SCOUT:
            # Symmetry-break: scout_0 owns casualty A, scout_1 owns casualty B.
            # Shared-policy MAPPO with identical observations made both scouts
            # converge on the same casualty (eval video iter 399 confirmed this).
            # Forcing distinct goal_coords per scout splits the gradient pull
            # and forces parallel discovery.
            target = agent.state.pos[idxs].clone()  # default: stay (idle)
            if agent.role_index == 0:
                # Scout 0: pull toward casualty A while it's unfound.
                unfound = (sa == SUB_STATE_INDEX[UNFOUND])
                target = torch.where(unfound.unsqueeze(-1), cas0_pos, target)
            else:
                # Scout 1 (and any further scouts): pull toward casualty B.
                unfound = (sb == SUB_STATE_INDEX[UNFOUND])
                target = torch.where(unfound.unsqueeze(-1), cas1_pos, target)
        else:
            # Carrier — phased target so the shaping gradient pulls them through
            # the loading sub-task, not just to the stretcher.
            stretcher_pos = self.stretcher.state.pos[idxs]
            found_a = (sa == SUB_STATE_INDEX[FOUND])
            found_b = (sb == SUB_STATE_INDEX[FOUND])
            loaded_a = (sa == SUB_STATE_INDEX[LOADED])
            loaded_b = (sb == SUB_STATE_INDEX[LOADED])

            # Phase 1: nothing found yet → stay near stretcher (ready to dock).
            target = stretcher_pos.clone()

            # Phase 2: a casualty is FOUND but not yet LOADED → target the
            # casualty itself, so carriers drag the stretcher there.
            target = torch.where(found_a.unsqueeze(-1), cas0_pos, target)
            target = torch.where(found_b.unsqueeze(-1), cas1_pos, target)
            # If both found, pick the one closer to the stretcher (faster pickup).
            both_found = found_a & found_b
            d_a = torch.linalg.vector_norm(stretcher_pos - cas0_pos, dim=-1)
            d_b = torch.linalg.vector_norm(stretcher_pos - cas1_pos, dim=-1)
            choose_b = both_found & (d_b < d_a)
            target = torch.where(choose_b.unsqueeze(-1), cas1_pos, target)

            # Phase 3: a casualty is LOADED → target its bay.
            target = torch.where(loaded_a.unsqueeze(-1), bay0_pos, target)
            target = torch.where(loaded_b.unsqueeze(-1), bay1_pos, target)

            # Phase 4: both delivered → idle.
            done = (sa == SUB_STATE_INDEX[DELIVERED]) & (sb == SUB_STATE_INDEX[DELIVERED])
            target = torch.where(done.unsqueeze(-1), agent.state.pos[idxs], target)

        if env_index is None:
            agent.goal_coords = target
        else:
            agent.goal_coords[env_index] = target.squeeze(0)

    # ================================================================== #
    #  Atomic-proposition detection (centralised, driven by world state)
    # ================================================================== #

    def _detect_event_vec(self) -> torch.Tensor:
        """Compute the per-env event vector (B, EVENT_DIM) from current world state.

        Used both to update the LanguageUnit's joint state at runtime and to
        shape rewards. Centralised observation is fine — only the *policy*
        is decentralised.
        """
        B = self.world.batch_dim
        device = self.world.device
        e = torch.zeros((B, self.event_dim), device=device)

        cas_pos = [c.state.pos for c in self.casualties]   # each (B, 2)
        stretch_pos = self.stretcher.state.pos
        bay_pos = [b.state.pos for b in self.bays]

        scouts = [a for a in self.world.agents if a.role == ROLE_SCOUT]
        carriers = [a for a in self.world.agents if a.role == ROLE_CARRIER]

        sa = self.language_unit.state_A
        sb = self.language_unit.state_B
        # Per-env DFA variant flags. The variant-id mapping mirrors
        # all_configs() in random_automaton_walk_search_rescue.py:
        #   0 = free_order            — no extra gates
        #   1 = constrained_a_first   — B's LOAD requires sa == DELIVERED
        #   2 = constrained_b_first   — A's LOAD requires sb == DELIVERED
        #   3 = heavy_strict_ab       — B's FOUND and B's LOAD both require sa == DELIVERED
        # Gating events here (rather than in _step_world_state) means reward
        # is also gated: the policy only earns r_discover/r_load when the
        # transition actually advances the variant's joint state.
        aut = self.language_unit.automaton_id
        is_a_first = (aut == 1)
        is_b_first = (aut == 2)
        is_heavy = (aut == 3)

        a_delivered = (sa == SUB_STATE_INDEX[DELIVERED])
        b_delivered = (sb == SUB_STATE_INDEX[DELIVERED])

        # Found events: any scout within detect_radius of an unfound casualty.
        # Variant gate: in heavy_strict, B can only become FOUND once A is delivered.
        for cas_idx, found_event in enumerate([CASUALTY_A_FOUND, CASUALTY_B_FOUND]):
            cur_state = sa if cas_idx == 0 else sb
            unfound = cur_state == SUB_STATE_INDEX[UNFOUND]
            if cas_idx == 0:
                gate = torch.ones_like(unfound)  # A discoverable in all variants
            else:
                gate = (~is_heavy) | a_delivered  # B gated in heavy
            if not (unfound & gate).any():
                continue
            min_d = torch.full((B,), float("inf"), device=device)
            for s in scouts:
                d = torch.linalg.vector_norm(s.state.pos - cas_pos[cas_idx], dim=-1)
                min_d = torch.minimum(min_d, d)
            e[:, found_event] = ((min_d <= self.detect_radius) & unfound & gate).float()

        # Carriers synced: ≥2 carriers within dock_radius of stretcher.
        if len(carriers) >= 2:
            close = torch.zeros((B, len(carriers)), device=device)
            for i, c in enumerate(carriers):
                close[:, i] = (torch.linalg.vector_norm(c.state.pos - stretch_pos, dim=-1) <= self.dock_radius).float()
            e[:, CARRIERS_SYNCED] = (close.sum(dim=-1) >= 2).float()

        # Stretcher loaded events: synced + stretcher near a found-but-not-loaded casualty.
        # Variant gates:
        #   STRETCHER_LOADED_A blocked in constrained_b_first until B is delivered.
        #   STRETCHER_LOADED_B blocked in (constrained_a_first OR heavy) until A is delivered.
        for cas_idx, load_event in enumerate([STRETCHER_LOADED_A, STRETCHER_LOADED_B]):
            cur_state = sa if cas_idx == 0 else sb
            other_state = sb if cas_idx == 0 else sa
            eligible = (cur_state == SUB_STATE_INDEX[FOUND]) & (other_state != SUB_STATE_INDEX[LOADED])
            if cas_idx == 0:
                gate = (~is_b_first) | b_delivered
            else:
                gate = (~(is_a_first | is_heavy)) | a_delivered
            d = torch.linalg.vector_norm(stretch_pos - cas_pos[cas_idx], dim=-1)
            e[:, load_event] = (eligible & gate & (d <= self.pickup_radius) & (e[:, CARRIERS_SYNCED] > 0.5)).float()

        # Delivered events: stretcher with cas_x loaded near bay_x. No variant gate
        # — once a casualty is LOADED, delivery is always allowed.
        for cas_idx, deliver_event, bay_idx in [
            (0, DELIVERED_A, 0), (1, DELIVERED_B, 1),
        ]:
            cur_state = sa if cas_idx == 0 else sb
            loaded = cur_state == SUB_STATE_INDEX[LOADED]
            d = torch.linalg.vector_norm(stretch_pos - bay_pos[bay_idx], dim=-1)
            e[:, deliver_event] = (loaded & (d <= self.bay_radius)).float()

        return e

    # ================================================================== #
    #  Per-step bookkeeping (called from reward — first agent only)
    # ================================================================== #

    def _step_world_state(self) -> torch.Tensor:
        """Detect events, advance the LanguageUnit's joint state, drag stretcher / casualty.

        Returns the event vector (B, EVENT_DIM) so reward terms can read it.
        """
        e = self._detect_event_vec()

        # Advance joint state in-place from the event vector + mutex.
        # Mirrors the single-step semantics in ParallelAutomaton.step.
        sa = self.language_unit.state_A.clone()
        sb = self.language_unit.state_B.clone()

        # A side
        sa = torch.where((sa == SUB_STATE_INDEX[UNFOUND]) & (e[:, CASUALTY_A_FOUND] > 0.5),
                         torch.full_like(sa, SUB_STATE_INDEX[FOUND]), sa)
        sa = torch.where(
            (sa == SUB_STATE_INDEX[FOUND]) & (e[:, STRETCHER_LOADED_A] > 0.5)
            & (e[:, CARRIERS_SYNCED] > 0.5) & (sb != SUB_STATE_INDEX[LOADED]),
            torch.full_like(sa, SUB_STATE_INDEX[LOADED]), sa,
        )
        sa = torch.where((sa == SUB_STATE_INDEX[LOADED]) & (e[:, DELIVERED_A] > 0.5),
                         torch.full_like(sa, SUB_STATE_INDEX[DELIVERED]), sa)
        # B side
        sb = torch.where((sb == SUB_STATE_INDEX[UNFOUND]) & (e[:, CASUALTY_B_FOUND] > 0.5),
                         torch.full_like(sb, SUB_STATE_INDEX[FOUND]), sb)
        sb = torch.where(
            (sb == SUB_STATE_INDEX[FOUND]) & (e[:, STRETCHER_LOADED_B] > 0.5)
            & (e[:, CARRIERS_SYNCED] > 0.5) & (sa != SUB_STATE_INDEX[LOADED]),
            torch.full_like(sb, SUB_STATE_INDEX[LOADED]), sb,
        )
        sb = torch.where((sb == SUB_STATE_INDEX[LOADED]) & (e[:, DELIVERED_B] > 0.5),
                         torch.full_like(sb, SUB_STATE_INDEX[DELIVERED]), sb)

        self.language_unit.state_A = sa
        self.language_unit.state_B = sb

        # Re-derive stretcher_carrying purely from current LOADED state.
        # Earlier we kept a "DELIVERED → -1" rule that fired every step after
        # the first delivery — once sa was DELIVERED, that rule kept clearing
        # the carrying flag even when sb later became LOADED, so casualty B
        # never got its position update and visually stayed at its spawn while
        # the team moved on. Mutex guarantees at most one of (sa,sb) is LOADED
        # at any time, so this assignment is unambiguous.
        carrying = torch.full_like(self.stretcher_carrying, -1)
        carrying = torch.where(sa == SUB_STATE_INDEX[LOADED], torch.zeros_like(carrying), carrying)
        carrying = torch.where(sb == SUB_STATE_INDEX[LOADED], torch.ones_like(carrying), carrying)
        self.stretcher_carrying = carrying

        # Stretcher physics: when synced, stretcher centroids on the docked carriers.
        carriers = [a for a in self.world.agents if a.role == ROLE_CARRIER]
        if len(carriers) >= 2 and (e[:, CARRIERS_SYNCED] > 0.5).any():
            stretch_pos = self.stretcher.state.pos
            new_pos = stretch_pos.clone()
            mask = e[:, CARRIERS_SYNCED] > 0.5
            close_pos = []
            for c in carriers:
                close = (torch.linalg.vector_norm(c.state.pos - stretch_pos, dim=-1) <= self.dock_radius)
                close_pos.append((close, c.state.pos))
            # Centroid of all carriers within dock_radius
            num = torch.zeros_like(stretch_pos)
            denom = torch.zeros((stretch_pos.shape[0],), device=stretch_pos.device)
            for close, pos in close_pos:
                num = num + close.unsqueeze(-1).float() * pos
                denom = denom + close.float()
            avg = torch.where(denom.unsqueeze(-1) > 0, num / denom.clamp(min=1).unsqueeze(-1), stretch_pos)
            new_pos = torch.where(mask.unsqueeze(-1), avg, stretch_pos)
            self.stretcher.set_pos(new_pos, batch_index=None)

        # Stretcher velocity: zero it every step. Without this, the next
        # world.step() integrates whatever residual velocity the stretcher
        # picked up from prior frames over 5 substeps × dt → drift away from
        # the centroid → carriers reach a casualty but the stretcher arrives
        # ~half a step late, missing pickup_radius. (User-visible symptom:
        # team passes over the second target without loading.)
        self.stretcher.state.vel = torch.zeros_like(self.stretcher.state.vel)

        # Loaded casualty follows the stretcher (and is also velocity-zeroed
        # so its dynamics don't drag it sideways while being "carried").
        for i, cas in enumerate(self.casualties):
            being_carried = (self.stretcher_carrying == i)
            if being_carried.any():
                cas.set_pos(
                    torch.where(being_carried.unsqueeze(-1), self.stretcher.state.pos, cas.state.pos),
                    batch_index=None,
                )
                cas.state.vel = torch.where(
                    being_carried.unsqueeze(-1),
                    torch.zeros_like(cas.state.vel),
                    cas.state.vel,
                )

        # Cache event vec for reward / observation reuse
        self._cached_event_vec = e
        return e

    # ================================================================== #
    #  Observation
    # ================================================================== #

    def observation(self, agent: Agent) -> torch.Tensor:
        B = self.world.batch_dim
        e = getattr(self, "_cached_event_vec", torch.zeros((B, self.event_dim), device=self.world.device))

        own_pos = agent.state.pos
        own_vel = agent.state.vel
        rel_cas0 = self.casualties[0].state.pos - own_pos
        rel_cas1 = self.casualties[1].state.pos - own_pos
        rel_stretcher = self.stretcher.state.pos - own_pos
        rel_bay0 = self.bays[0].state.pos - own_pos
        rel_bay1 = self.bays[1].state.pos - own_pos
        return torch.cat([
            own_pos, own_vel, agent.role_one_hot,
            rel_cas0, rel_cas1, rel_stretcher, rel_bay0, rel_bay1, e,
        ], dim=-1)

    # ================================================================== #
    #  Reward
    # ================================================================== #

    def reward(self, agent: Agent) -> torch.Tensor:
        is_first = agent == self.world.agents[0]
        device = self.world.device
        B = self.world.batch_dim

        if is_first:
            # Step world state once per env step — invoked from the first agent.
            self._step_world_state()
            self._goal_coords_recomputed = False

        e = self._cached_event_vec
        # Refresh goal coords now that the joint state has potentially changed.
        if not getattr(self, "_goal_coords_recomputed", True):
            for a in self.world.agents:
                self._set_goal_coords(a, env_index=None)
            self._goal_coords_recomputed = True

        rew = torch.zeros(B, device=device)

        # Distance shaping. For scouts: own_pos → goal (find a casualty).
        # For carriers: STRETCHER → goal (drag stretcher to casualty/bay) plus
        # a small carrier→stretcher term (stay docked). The previous "carrier
        # → goal" formulation pulled each carrier individually toward the
        # casualty, leaving the stretcher behind — load events never fired.
        shaping = self.shaping_factor_scout if agent.role == ROLE_SCOUT else self.shaping_factor_carrier
        if agent.role == ROLE_SCOUT:
            d = torch.linalg.vector_norm(agent.state.pos - agent.goal_coords, dim=-1)
        else:
            d_main = torch.linalg.vector_norm(self.stretcher.state.pos - agent.goal_coords, dim=-1)
            d_dock = torch.linalg.vector_norm(agent.state.pos - self.stretcher.state.pos, dim=-1)
            # Heavy dock weight (3.0): carriers must stay tethered to the
            # stretcher. Previous 0.3 weight was too weak — eval video showed
            # carriers abandoning the stretcher to follow scouts to casualties.
            d = d_main + 3.0 * d_dock
        # Zero out shaping once mission complete so policy can finish cleanly.
        sa_now = self.language_unit.state_A
        sb_now = self.language_unit.state_B
        not_done = ~((sa_now == SUB_STATE_INDEX[DELIVERED]) & (sb_now == SUB_STATE_INDEX[DELIVERED]))
        rew = rew - shaping * d * 0.001 * not_done.float()

        # Sparse role-conditioned bonuses on event firings
        if agent.role == ROLE_SCOUT:
            rew = rew + self.r_discover * (e[:, CASUALTY_A_FOUND] + e[:, CASUALTY_B_FOUND])
        else:
            rew = rew + self.r_load * (e[:, STRETCHER_LOADED_A] + e[:, STRETCHER_LOADED_B])
            rew = rew + self.r_deliver * (e[:, DELIVERED_A] + e[:, DELIVERED_B])
            # Sync bonus only when at least one casualty is in FOUND (we want pickup soon)
            sa = self.language_unit.state_A
            sb = self.language_unit.state_B
            either_found = (sa == SUB_STATE_INDEX[FOUND]) | (sb == SUB_STATE_INDEX[FOUND])
            rew = rew + self.r_sync_step * (e[:, CARRIERS_SYNCED] * either_found.float())

        # Terminal bonus on full DFA acceptance
        sa = self.language_unit.state_A
        sb = self.language_unit.state_B
        terminal = (sa == SUB_STATE_INDEX[DELIVERED]) & (sb == SUB_STATE_INDEX[DELIVERED])
        rew = rew + self.r_terminal * terminal.float() * (1.0 / max(1, self.n_agents))

        return rew

    # ================================================================== #
    #  Action processing — runs the velocity controller
    # ================================================================== #

    def process_action(self, agent: Agent) -> None:
        """Convert the policy's target-velocity action into a force command.

        VMAS's BaseScenario.process_action is a no-op by default; without
        this override the velocity controller is never invoked, action.f
        stays zero, and agents don't move at all (silent failure during
        both training and evaluation). Mirrors four_flags's process_action.
        """
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.005] = 0
        if self.use_velocity_controller:
            agent.controller.process_force()

    # ================================================================== #
    #  Done
    # ================================================================== #

    def done(self) -> torch.Tensor:
        sa = self.language_unit.state_A
        sb = self.language_unit.state_B
        return (sa == SUB_STATE_INDEX[DELIVERED]) & (sb == SUB_STATE_INDEX[DELIVERED])
