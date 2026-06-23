"""Episode runner for the online-LLM-planner baseline.

One episode at a time, batch_dim=1, replanning on joint-state change or every
``replan_every`` steps. Records every event (joint-state transition) and the
per-step action assignments for later inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from sequence_models.search_rescue.random_automaton_walk_search_rescue import (
    SUB_STATES, UNFOUND, FOUND, LOADED, DELIVERED, SUB_STATE_INDEX,
    CASUALTY_A_FOUND, CASUALTY_B_FOUND,
    STRETCHER_LOADED_A, STRETCHER_LOADED_B,
    DELIVERED_A, DELIVERED_B, CARRIERS_SYNCED,
)
from evaluations.baselines.online_llm_planner.planner import (
    OnlineLLMPlanner, PlanResult, CumulativePlannerStats,
)
from evaluations.baselines.online_llm_planner.prompts import format_world_state
from evaluations.baselines.online_llm_planner.controller import actions_to_velocities

EVENT_NAMES = {
    CASUALTY_A_FOUND: "casualty_A_found",
    CASUALTY_B_FOUND: "casualty_B_found",
    STRETCHER_LOADED_A: "stretcher_loaded_A",
    STRETCHER_LOADED_B: "stretcher_loaded_B",
    DELIVERED_A: "delivered_A",
    DELIVERED_B: "delivered_B",
    CARRIERS_SYNCED: "carriers_synced",
}


@dataclass
class EpisodeRecord:
    summary: str
    automaton_name: str
    success: bool
    n_steps: int
    final_joint_state: tuple
    events_log: List[str]
    plan_log: List[Dict]
    planner_stats: Dict
    failure_reason: Optional[str] = None


def _joint_state(scenario, env_idx: int = 0) -> tuple[str, str]:
    return (
        SUB_STATES[int(scenario.language_unit.state_A[env_idx])],
        SUB_STATES[int(scenario.language_unit.state_B[env_idx])],
    )


def _set_summary(scenario, summary_text: str, env_idx: int = 0) -> None:
    """Inject the LLM-readable summary into the scenario's language unit.

    The LLM doesn't actually use the embedding (`y`) from the LanguageUnit —
    it reads `summary` directly — but we set the summary string so other
    code (e.g. extra_render) sees the right text.
    """
    scenario.language_unit.summary[env_idx] = summary_text


def run_one_episode(
    *,
    scenario,
    planner: OnlineLLMPlanner,
    summary: str,
    automaton_name: str,
    max_steps: int = 300,
    replan_every: int = 20,
    v_max: float = 1.0,
    verbose: bool = False,
) -> EpisodeRecord:
    """Run one episode at batch_dim=1.

    ``scenario`` must already be constructed with batch_dim=1 and have its
    ``_world`` attribute set (see evaluate.py for the construction code).
    """
    scenario.reset_world_at(env_index=None)
    _set_summary(scenario, summary)

    events_log: List[str] = []
    plan_log: List[Dict] = []
    stats = CumulativePlannerStats()

    last_joint_state = _joint_state(scenario)
    last_plan_step = -10**9
    current_actions: Optional[List[str]] = None
    failure_reason: Optional[str] = None

    for step in range(max_steps):
        joint = _joint_state(scenario)
        joint_changed = (joint != last_joint_state)

        if current_actions is None or joint_changed or (step - last_plan_step) >= replan_every:
            # Replan
            world_text = _world_state_text(scenario, summary, step, max_steps, events_log)
            try:
                result = planner.plan(world_text)
            except Exception as exc:
                stats.failures += 1
                failure_reason = f"planner failure at step {step}: {exc}"
                break
            stats.add(result)
            current_actions = [result.actions[f"agent_{i}"] for i in range(4)]
            last_plan_step = step
            plan_log.append({
                "step": step,
                "joint_state": list(joint),
                "actions": current_actions,
                "reasoning": result.reasoning,
                "latency_s": round(result.latency_s, 3),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            })
            if verbose:
                print(f"  [step {step}] joint={joint} → actions={current_actions} | {result.reasoning}")

        if joint_changed:
            events_log.append(f"step {step}: joint_state {last_joint_state} → {joint}")
        last_joint_state = joint

        # Translate actions to velocities and step the world.
        velocities = actions_to_velocities(
            scenario=scenario,
            actions_per_agent=current_actions,
            env_index=0,
            v_max=v_max,
        )
        for i, agent in enumerate(scenario.world.agents):
            # Action tensor is shape (B, 2); we have B=1.
            agent.action.u = velocities[i].unsqueeze(0)
        # IMPORTANT: replicate what vmas Environment.env_process_action does:
        # (1) scenario.process_action — our PID converts target velocity to force
        # (2) agent.dynamics.check_and_process_action — copies the post-PID
        #     action.u into agent.state.force, which is what world.step actually
        #     reads (we found this the hard way; without (2), agents stay put).
        for agent in scenario.world.agents:
            scenario.process_action(agent)
            agent.dynamics.check_and_process_action()
        scenario.world.step()

        # Run reward path so the scenario advances joint state + stretcher physics.
        for agent in scenario.world.agents:
            scenario.reward(agent)

        # Pull events that fired this step into the log
        ev = scenario._cached_event_vec[0].tolist()
        for idx, fired in enumerate(ev):
            if fired > 0.5 and idx in EVENT_NAMES and idx != CARRIERS_SYNCED:
                tag = f"step {step}: event {EVENT_NAMES[idx]}"
                # de-dup consecutive identical events
                if not events_log or events_log[-1] != tag:
                    events_log.append(tag)

        if scenario.done()[0]:
            return EpisodeRecord(
                summary=summary,
                automaton_name=automaton_name,
                success=True,
                n_steps=step + 1,
                final_joint_state=_joint_state(scenario),
                events_log=events_log,
                plan_log=plan_log,
                planner_stats=stats.summary(),
            )

    return EpisodeRecord(
        summary=summary,
        automaton_name=automaton_name,
        success=False,
        n_steps=max_steps,
        final_joint_state=_joint_state(scenario),
        events_log=events_log,
        plan_log=plan_log,
        planner_stats=stats.summary(),
        failure_reason=failure_reason or "timeout",
    )


def _world_state_text(scenario, summary: str, step: int, max_steps: int, events_log: List[str]) -> str:
    n = len(scenario.world.agents)
    agent_pos = [scenario.world.agents[i].state.pos[0].tolist() for i in range(n)]
    agent_vel = [scenario.world.agents[i].state.vel[0].tolist() for i in range(n)]
    # Roles: scouts first, then carriers (matches scenario.add_agents order).
    agent_roles = [
        "scout" if i < scenario.n_scouts else "carrier" for i in range(n)
    ]
    cas_pos = [scenario.casualties[i].state.pos[0].tolist() for i in range(2)]
    cas_states = [
        SUB_STATES[int(scenario.language_unit.state_A[0])],
        SUB_STATES[int(scenario.language_unit.state_B[0])],
    ]
    return format_world_state(
        summary=summary,
        step=step,
        max_steps=max_steps,
        agent_positions=agent_pos,
        agent_velocities=agent_vel,
        agent_roles=agent_roles,
        casualty_positions=cas_pos,
        casualty_states=cas_states,
        stretcher_position=scenario.stretcher.state.pos[0].tolist(),
        bay_positions=[scenario.bays[i].state.pos[0].tolist() for i in range(2)],
        recent_events=events_log,
    )
