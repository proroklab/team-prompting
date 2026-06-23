"""Parallel-automaton machinery for the Search-and-Rescue task.

Extends the single-automaton DFA framework in
``sequence_models/four_flags/random_automaton_walk.py`` to support:

* parallel sub-automata that share a single event vector,
* cross-state guards (transitions that read the full joint state),
* a single-stretcher mutex enforced at composition time,
* random-walk data generation honouring physical exclusion constraints.

Three prompt families produce four DFA variants the RNN must encode jointly:

1. ``free_order``                     — either casualty first.
2. ``constrained_a_first``            — A must be rescued before B.
3. ``constrained_b_first``            — symmetric.
4. ``heavy_casualty_strict_AB``       — strict sequential, no parallel region.

The output JSON schema matches the existing four_flags pipeline so the same
``EventSequenceDataset`` / ``EventRNN`` training code can consume it.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Global experiment parameters
# --------------------------------------------------------------------------- #

MIN_SEQ_STEPS: int = 4
MAX_SEQ_STEPS: int = 20
REPS: int = 500
RNG_SEED: int | None = 42

# --------------------------------------------------------------------------- #
#  Event vector
# --------------------------------------------------------------------------- #

EVENT_DIM: int = 7

CASUALTY_A_FOUND, CASUALTY_B_FOUND = 0, 1
STRETCHER_LOADED_A, STRETCHER_LOADED_B = 2, 3
DELIVERED_A, DELIVERED_B = 4, 5
CARRIERS_SYNCED = 6

EVENTS: Sequence[str] = (
    "casualty_A_found",
    "casualty_B_found",
    "stretcher_loaded_A",
    "stretcher_loaded_B",
    "delivered_A",
    "delivered_B",
    "carriers_synced",
)

# --------------------------------------------------------------------------- #
#  Sub-states (4 per pipeline, two pipelines → 8 sub-state bits total)
# --------------------------------------------------------------------------- #

UNFOUND, FOUND, LOADED, DELIVERED = "unfound", "found", "loaded", "delivered"
SUB_STATES: Sequence[str] = (UNFOUND, FOUND, LOADED, DELIVERED)
N_SUB = 2  # two casualty pipelines
SUB_STATE_DIM = len(SUB_STATES)  # 4

# Flat sub-state-bit index in the concatenated [A | B] one-hot.
SUB_STATE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(SUB_STATES)}

# --------------------------------------------------------------------------- #
#  Configuration: number of automata in this dataset.
#
#  Set lazily at __main__; the state-encoding helper uses NUM_AUTOMATA below.
# --------------------------------------------------------------------------- #

NUM_AUTOMATA: int = 4  # free, constrained_A, constrained_B, heavy_strict_AB


def _automaton_id_bits(automaton_id: int, num_automata: int = NUM_AUTOMATA) -> List[int]:
    """Binary encoding of the automaton id, MSB first.

    Width is ``ceil(log2(num_automata))`` (≥1 bit). Mirrors the convention
    used in ``four_flags/random_automaton_walk.py``.
    """
    width = max(1, math.ceil(math.log2(max(2, num_automata))))
    return [(automaton_id >> i) & 1 for i in reversed(range(width))]


def joint_state_to_one_hot(
    joint_state: Tuple[str, str],
    automaton_id: int,
    num_automata: int = NUM_AUTOMATA,
) -> List[int]:
    """Encode a joint state as ``[automaton_id_bits | A_one_hot | B_one_hot]``.

    Total length = ceil(log2(num_automata)) + 2 * SUB_STATE_DIM.
    """
    sub_one_hot: List[int] = [0] * (N_SUB * SUB_STATE_DIM)
    for sub_idx, sub_state in enumerate(joint_state):
        sub_one_hot[sub_idx * SUB_STATE_DIM + SUB_STATE_INDEX[sub_state]] = 1
    return _automaton_id_bits(automaton_id, num_automata) + sub_one_hot


# Total state-output dimensionality for the EventRNN head. We default to the
# maximum-supported ``num_automata`` (4) so STATE_DIM is stable across configs.
STATE_DIM: int = max(1, math.ceil(math.log2(max(2, NUM_AUTOMATA)))) + N_SUB * SUB_STATE_DIM


# --------------------------------------------------------------------------- #
#  Predicates over (event_vec, joint_state)
# --------------------------------------------------------------------------- #

Vector = List[int]
JointState = Tuple[str, str]
Pred = Callable[[Vector, JointState], bool]


def has(bit_idx: int) -> Pred:
    return lambda v, _js, i=bit_idx: bool(v[i])


def not_has(bit_idx: int) -> Pred:
    return lambda v, _js, i=bit_idx: not v[i]


def all_of(*preds: Pred) -> Pred:
    return lambda v, js, ps=preds: all(p(v, js) for p in ps)


def any_of(*preds: Pred) -> Pred:
    return lambda v, js, ps=preds: any(p(v, js) for p in ps)


def other_in(other_sub_idx: int, *states: str) -> Pred:
    """True iff sub-automaton ``other_sub_idx`` is currently in any of ``states``."""
    return lambda _v, js, oi=other_sub_idx, ss=set(states): js[oi] in ss


def other_not_in(other_sub_idx: int, *states: str) -> Pred:
    return lambda _v, js, oi=other_sub_idx, ss=set(states): js[oi] not in ss


def advance_if(pred: Pred, then_state: str, else_state: str) -> Callable[[Vector, JointState], str]:
    return lambda v, js, p=pred, t=then_state, e=else_state: (t if p(v, js) else e)


def advance_case(*cases: Tuple[Pred, str], default: str) -> Callable[[Vector, JointState], str]:
    """Ordered guards: first matching ``(pred, state)`` wins; else ``default``."""
    def step(v: Vector, js: JointState, cs=cases, d=default) -> str:
        for pred, st in cs:
            if pred(v, js):
                return st
        return d
    return step


# --------------------------------------------------------------------------- #
#  Sub-automaton + parallel composition
# --------------------------------------------------------------------------- #

class SubAutomaton:
    """A single casualty-pipeline automaton.

    ``transition[s](event_vec, joint_state) -> next_state`` may consult
    other sub-automata's states via the joint state.
    """

    def __init__(
        self,
        idx: int,
        transition: Dict[str, Callable[[Vector, JointState], str]],
        initial: str,
    ) -> None:
        self.idx = idx
        self._transition = transition
        self.initial = initial

    def step(self, joint_state: JointState, event_vec: Vector) -> str:
        cur = joint_state[self.idx]
        return self._transition[cur](event_vec, joint_state)


class ParallelAutomaton:
    """Two casualty pipelines stepped synchronously from the same prior state.

    All sub-automata read the *prior* joint state, then commit their next
    states simultaneously. A static mutex post-condition rejects illegal joint
    states (defence in depth — the data generator already excludes the
    physically impossible event combinations that would produce them).
    """

    def __init__(
        self,
        subs: List[SubAutomaton],
        automaton_id: int,
        finals: Callable[[JointState], bool],
        num_automata: int = NUM_AUTOMATA,
    ) -> None:
        assert len(subs) == N_SUB
        self._subs = subs
        self.automaton_id = automaton_id
        self._finals = finals
        self._num_automata = num_automata

    @property
    def initial(self) -> JointState:
        return tuple(s.initial for s in self._subs)  # type: ignore[return-value]

    def step(self, joint_state: JointState, event_vec: Vector) -> JointState:
        next_state = tuple(sub.step(joint_state, event_vec) for sub in self._subs)
        # Mutex: at most one pipeline in LOADED at a time. If somehow violated
        # (caller fed an unconstrained event vector), revert the second sub
        # to its prior state — a deterministic tie-break.
        if sum(1 for s in next_state if s == LOADED) > 1:
            next_state = (next_state[0], joint_state[1])
        return next_state  # type: ignore[return-value]

    def run(self, event_vecs: Sequence[Vector]) -> List[Tuple[Vector | None, JointState, List[int]]]:
        joint = self.initial
        trace: List[Tuple[Vector | None, JointState, List[int]]] = [
            (None, joint, joint_state_to_one_hot(joint, self.automaton_id, self._num_automata))
        ]
        for v in event_vecs:
            joint = self.step(joint, v)
            trace.append(
                (list(v), joint, joint_state_to_one_hot(joint, self.automaton_id, self._num_automata))
            )
        return trace

    def is_final(self, joint_state: JointState) -> bool:
        return self._finals(joint_state)


# --------------------------------------------------------------------------- #
#  Concrete DFA configs (the three prompt families → four variants)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DFAConfig:
    name: str
    automaton_id: int
    builder: Callable[[int], ParallelAutomaton]
    outfile: Path


def _pipeline_transition(
    sub_idx: int,
    found_event: int,
    loaded_event: int,
    delivered_event: int,
    *,
    extra_loaded_guard: Pred | None = None,
) -> Dict[str, Callable[[Vector, JointState], str]]:
    """Build the four-state transition table for one casualty pipeline.

    States: ``unfound → found → loaded → delivered``.
    Transitions:
      - unfound → found     on ``found_event``
      - found → loaded      on ``loaded_event AND carriers_synced AND other_not_in(LOADED)``
                            optionally further gated by ``extra_loaded_guard``
      - loaded → delivered  on ``delivered_event``
      - delivered           absorbing
    """
    other_idx = 1 - sub_idx
    base_loaded_guard = all_of(
        has(loaded_event),
        has(CARRIERS_SYNCED),
        other_not_in(other_idx, LOADED),
    )
    loaded_guard: Pred = (
        base_loaded_guard
        if extra_loaded_guard is None
        else all_of(base_loaded_guard, extra_loaded_guard)
    )

    return {
        UNFOUND:   advance_if(has(found_event),     FOUND,     UNFOUND),
        FOUND:     advance_if(loaded_guard,         LOADED,    FOUND),
        LOADED:    advance_if(has(delivered_event), DELIVERED, LOADED),
        DELIVERED: (lambda _v, _js: DELIVERED),
    }


def _build_terminal_finals() -> Callable[[JointState], bool]:
    return lambda js: js[0] == DELIVERED and js[1] == DELIVERED


def free_order_builder(automaton_id: int) -> ParallelAutomaton:
    sub_a = SubAutomaton(
        idx=0,
        transition=_pipeline_transition(0, CASUALTY_A_FOUND, STRETCHER_LOADED_A, DELIVERED_A),
        initial=UNFOUND,
    )
    sub_b = SubAutomaton(
        idx=1,
        transition=_pipeline_transition(1, CASUALTY_B_FOUND, STRETCHER_LOADED_B, DELIVERED_B),
        initial=UNFOUND,
    )
    return ParallelAutomaton([sub_a, sub_b], automaton_id, _build_terminal_finals())


def constrained_a_first_builder(automaton_id: int) -> ParallelAutomaton:
    """A must be DELIVERED before B can be LOADED."""
    sub_a = SubAutomaton(
        idx=0,
        transition=_pipeline_transition(0, CASUALTY_A_FOUND, STRETCHER_LOADED_A, DELIVERED_A),
        initial=UNFOUND,
    )
    sub_b = SubAutomaton(
        idx=1,
        transition=_pipeline_transition(
            1, CASUALTY_B_FOUND, STRETCHER_LOADED_B, DELIVERED_B,
            extra_loaded_guard=other_in(0, DELIVERED),
        ),
        initial=UNFOUND,
    )
    return ParallelAutomaton([sub_a, sub_b], automaton_id, _build_terminal_finals())


def constrained_b_first_builder(automaton_id: int) -> ParallelAutomaton:
    sub_a = SubAutomaton(
        idx=0,
        transition=_pipeline_transition(
            0, CASUALTY_A_FOUND, STRETCHER_LOADED_A, DELIVERED_A,
            extra_loaded_guard=other_in(1, DELIVERED),
        ),
        initial=UNFOUND,
    )
    sub_b = SubAutomaton(
        idx=1,
        transition=_pipeline_transition(1, CASUALTY_B_FOUND, STRETCHER_LOADED_B, DELIVERED_B),
        initial=UNFOUND,
    )
    return ParallelAutomaton([sub_a, sub_b], automaton_id, _build_terminal_finals())


def heavy_strict_ab_builder(automaton_id: int) -> ParallelAutomaton:
    """Heavy casualty: strict A→B sequence, plus B can only be DISCOVERED after A is DELIVERED.

    This is the degenerate sanity case — no parallel exploration permitted.
    """
    sub_a = SubAutomaton(
        idx=0,
        transition=_pipeline_transition(0, CASUALTY_A_FOUND, STRETCHER_LOADED_A, DELIVERED_A),
        initial=UNFOUND,
    )
    # B's discovery is itself gated on A being delivered.
    b_transition: Dict[str, Callable[[Vector, JointState], str]] = {
        UNFOUND: advance_if(
            all_of(has(CASUALTY_B_FOUND), other_in(0, DELIVERED)),
            FOUND,
            UNFOUND,
        ),
        FOUND: advance_if(
            all_of(
                has(STRETCHER_LOADED_B),
                has(CARRIERS_SYNCED),
                other_in(0, DELIVERED),
                other_not_in(0, LOADED),
            ),
            LOADED,
            FOUND,
        ),
        LOADED: advance_if(has(DELIVERED_B), DELIVERED, LOADED),
        DELIVERED: (lambda _v, _js: DELIVERED),
    }
    sub_b = SubAutomaton(idx=1, transition=b_transition, initial=UNFOUND)
    return ParallelAutomaton([sub_a, sub_b], automaton_id, _build_terminal_finals())


# --------------------------------------------------------------------------- #
#  Random-walk generator with physical-exclusion constraints
# --------------------------------------------------------------------------- #

def _physically_consistent_event_vec(joint_state: JointState) -> Vector:
    """Sample one event vector that respects physical mutual exclusions.

    Constraints encoded:
      - At most one of ``stretcher_loaded_A`` / ``stretcher_loaded_B`` per step.
      - ``stretcher_loaded_X`` is only sampled when sub-X is in FOUND
        (and the other sub is not in LOADED).
      - ``delivered_X`` is only sampled when sub-X is in LOADED.
      - ``casualty_X_found`` is only sampled when sub-X is in UNFOUND.
      - ``carriers_synced`` is independent.
    Each eligible bit fires with probability 0.5 — gives a healthy mix of
    "transition fires" vs "self-loop" so the RNN sees both.
    """
    v: Vector = [0] * EVENT_DIM
    sa, sb = joint_state

    if sa == UNFOUND:
        v[CASUALTY_A_FOUND] = random.randint(0, 1)
    if sb == UNFOUND:
        v[CASUALTY_B_FOUND] = random.randint(0, 1)

    eligible_load: List[int] = []
    if sa == FOUND and sb != LOADED:
        eligible_load.append(STRETCHER_LOADED_A)
    if sb == FOUND and sa != LOADED:
        eligible_load.append(STRETCHER_LOADED_B)
    if eligible_load and random.random() < 0.5:
        v[random.choice(eligible_load)] = 1

    if sa == LOADED:
        v[DELIVERED_A] = random.randint(0, 1)
    if sb == LOADED:
        v[DELIVERED_B] = random.randint(0, 1)

    v[CARRIERS_SYNCED] = random.randint(0, 1)
    return v


def random_walk(automaton: ParallelAutomaton, steps: int) -> List[Tuple[Vector | None, JointState, List[int]]]:
    joint = automaton.initial
    vecs: List[Vector] = []
    for _ in range(steps):
        v = _physically_consistent_event_vec(joint)
        vecs.append(v)
        # Walk forward so the *next* sample is conditioned on the *new* state.
        joint = automaton.step(joint, v)
    return automaton.run(vecs)


def record_trace(trace: List[Tuple[Vector | None, JointState, List[int]]]) -> dict:
    events = [vec for (vec, _, _) in trace[1:]]
    states = [list(st) for (_, st, _) in trace]
    label = [lb for (_, _, lb) in trace]
    return {"events": events, "states": states, "label": label}


# --------------------------------------------------------------------------- #
#  Reachability check (sanity helper, not used at training time)
# --------------------------------------------------------------------------- #

def reachable_joint_states(automaton: ParallelAutomaton, max_iters: int = 1024) -> set[JointState]:
    """BFS over physically consistent event-vec choices to enumerate reachable states."""
    seen: set[JointState] = {automaton.initial}
    frontier: List[JointState] = [automaton.initial]
    iters = 0
    while frontier and iters < max_iters:
        iters += 1
        next_frontier: List[JointState] = []
        for js in frontier:
            for vec in _enumerate_event_vecs(js):
                ns = automaton.step(js, vec)
                if ns not in seen:
                    seen.add(ns)
                    next_frontier.append(ns)
        frontier = next_frontier
    return seen


def _enumerate_event_vecs(joint_state: JointState) -> List[Vector]:
    """All physically consistent event vectors at ``joint_state`` (small enumeration)."""
    sa, sb = joint_state
    bits_options: List[List[int]] = []
    bits_options.append([0, 1] if sa == UNFOUND else [0])           # CASUALTY_A_FOUND
    bits_options.append([0, 1] if sb == UNFOUND else [0])           # CASUALTY_B_FOUND
    bits_options.append([0, 1] if (sa == FOUND and sb != LOADED) else [0])  # LOADED_A
    bits_options.append([0, 1] if (sb == FOUND and sa != LOADED) else [0])  # LOADED_B
    bits_options.append([0, 1] if sa == LOADED else [0])            # DELIVERED_A
    bits_options.append([0, 1] if sb == LOADED else [0])            # DELIVERED_B
    bits_options.append([0, 1])                                     # CARRIERS_SYNCED

    out: List[Vector] = []
    def rec(prefix: List[int], i: int) -> None:
        if i == EVENT_DIM:
            # Filter: at most one stretcher-load event per step.
            if prefix[STRETCHER_LOADED_A] + prefix[STRETCHER_LOADED_B] <= 1:
                out.append(prefix.copy())
            return
        for b in bits_options[i]:
            prefix.append(b)
            rec(prefix, i + 1)
            prefix.pop()
    rec([], 0)
    return out


# --------------------------------------------------------------------------- #
#  DFA registry
# --------------------------------------------------------------------------- #

DATA_DIR = Path("sequence_models/data/search_rescue/random_walks")


def _cfg(name: str, automaton_id: int, builder) -> DFAConfig:
    return DFAConfig(
        name=name,
        automaton_id=automaton_id,
        builder=builder,
        outfile=DATA_DIR / f"random_walk_{name}.json",
    )


def all_configs() -> List[DFAConfig]:
    return [
        _cfg("free_order",            0, free_order_builder),
        _cfg("constrained_a_first",   1, constrained_a_first_builder),
        _cfg("constrained_b_first",   2, constrained_b_first_builder),
        _cfg("heavy_strict_ab",       3, heavy_strict_ab_builder),
    ]


# --------------------------------------------------------------------------- #
#  Master runner
# --------------------------------------------------------------------------- #

def run_and_save(cfg: DFAConfig, num_automata: int) -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED + cfg.automaton_id)

    dfa = cfg.builder(cfg.automaton_id)
    dfa._num_automata = num_automata  # ensure encoding width matches dataset

    runs: List[dict] = []
    for _ in range(REPS):
        L = random.randint(MIN_SEQ_STEPS, MAX_SEQ_STEPS)
        trace = random_walk(dfa, L)
        runs.append(record_trace(trace))

    cfg.outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg.outfile.write_text(json.dumps(runs, indent=2))
    print(f"[{cfg.name:>22s}] wrote {len(runs)} runs ➜ {cfg.outfile}")

    reach = reachable_joint_states(dfa)
    final_count = sum(1 for js in reach if dfa.is_final(js))
    print(f"[{cfg.name:>22s}] reachable joint states: {len(reach)} (final: {final_count})")


if __name__ == "__main__":
    cfgs = all_configs()
    NUM_AUTOMATA = len(cfgs)
    print(f"NUM_AUTOMATA = {NUM_AUTOMATA}, EVENT_DIM = {EVENT_DIM}, STATE_DIM = {STATE_DIM}")
    for cfg in cfgs:
        run_and_save(cfg, num_automata=NUM_AUTOMATA)
