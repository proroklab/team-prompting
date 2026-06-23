"""Build the augmented dataset for the Search-and-Rescue task.

Reads the raw random-walk JSONs produced by
``random_automaton_walk_search_rescue.py`` and attaches LLM-authored
natural-language descriptions + 1024-dim embeddings, producing a single merged
dataset that ``scenarios/search_rescue/language.py`` can load.

Schema matches what the EventRNN training pipeline expects:

  per trace:
    events  : List[List[int]]     (T × EVENT_DIM)            — from random walk
    states  : List[List[str]]     (T+1) × N_SUB              — from random walk
    label   : List[List[int]]     (T+1) × STATE_DIM          — from random walk
    responses : List[str]         (T+1)                      — sub-task per step
    h         : List[List[float]] (T+1) × 1024               — per-step embedding
    summary   : str                                          — high-level task
    y         : List[float]       1024                       — summary embedding
    subtask_decoder_labels : List[List[int]] (T+1) × subtask_dim   (zeros for now)

Templates are intentionally varied along several axes (lexicon, framing,
ordering, specificity) so the trained RNN learns paraphrase invariance — one
of the explicit reviewer asks for the revision.

Run:
    PYTHONPATH=. python -m sequence_models.search_rescue.build_augmented_dataset_search_rescue
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

EMBEDDING_MODEL = "thenlper/gte-large"  # 1024-dim, matches existing four_flags pipeline
EMBEDDING_DIM = 1024
RNG_SEED = 13
SUBTASK_DECODER_DIM = 8  # placeholder; matches the per-step decoder label width

DATA_DIR = Path("sequence_models/data/search_rescue")
WALKS_DIR = DATA_DIR / "random_walks"
OUT_PATH = DATA_DIR / "dataset_search_rescue_ALL.json"

# Map DFA file stem → automaton family used by the templates below.
DFA_FILES: List[Tuple[str, str]] = [
    ("free_order",          "random_walk_free_order.json"),
    ("constrained_a_first", "random_walk_constrained_a_first.json"),
    ("constrained_b_first", "random_walk_constrained_b_first.json"),
    ("heavy_strict_ab",     "random_walk_heavy_strict_ab.json"),
]

# --------------------------------------------------------------------------- #
#  High-level task summaries
#
#  Every entry is a complete utterance. Variation axes:
#   * lexicon: casualty / victim / injured / survivor / patient
#   * framing: imperative / declarative / narrative
#   * ordering: mission-first vs. constraint-first
#   * specificity: detailed bay-color cues vs. abstract goal
# --------------------------------------------------------------------------- #

SUMMARIES_FREE_ORDER: List[str] = [
    "There are two casualties in the area. Find both and bring each one to a medical bay.",
    "Locate the two injured people and transport them to the appropriate medical bays.",
    "Two victims need rescuing. Search the area, then carry each to a bay.",
    "Find both casualties and deliver them to the medical bays — order does not matter.",
    "There are two survivors out there; the team must find them and move them to safety.",
    "Rescue both injured persons and bring each to a medical bay.",
    "Search the warehouse for two casualties and deliver them to the medical posts.",
    "Two patients are stranded in the area. Carry them out to the bays.",
    "Mission: locate the two victims and transport them to the bays. Pick whichever you find first.",
    "We have two people down. Find them and bring them to the medical area.",
    "Spread out, scan the floor for two casualties, then move each to a bay using the stretcher.",
    "Two injured people are scattered in the warehouse. Get them both to safety.",
    "There are two survivors waiting for evacuation. Find them and get them to the medical bays.",
    "Recover both casualties from the search area and deliver them to medical.",
    "Two people need help. Find them and carry each one to a bay — no specific order.",
    "Scout the room for the two victims and have the carriers transport each to a medical bay.",
    "Two casualties are unaccounted for. Locate and evacuate both.",
    "Patients are down in the warehouse. Find both and bring them out.",
    "Find both injured people and transport them to the bays. Either order is fine.",
    "Two people are hurt. The team needs to find both and move each to a medical bay.",
    "We need to evacuate two casualties from the area. Free choice of order.",
    "Locate two victims in the warehouse, then carry them to the bays.",
    "Two survivors are waiting. Search the area and move them to the medical posts.",
    "Two casualties are inside. Find both and transport them to the medical zones.",
    "Two injured individuals must be rescued. Get them to the bays in any order.",
    "Sweep the area for two patients and bring them to the bays.",
    "Two people need extraction. Find them and carry each to a bay.",
    "Recover the two casualties and bring them to the medical bays in whatever order works.",
    "Get both victims out of the warehouse and into a medical bay.",
    "Mission objective: rescue two casualties. Bring each to a bay.",
]

SUMMARIES_CONSTRAINED_A_FIRST: List[str] = [
    "Rescue the red casualty first; only after they are at the green bay should you go for the orange one.",
    "Priority is the red casualty. Save them first. Then deal with the orange one.",
    "There are two injured people. Do the red one before the orange one.",
    "Save the red casualty before the orange casualty.",
    "Red victim first, then orange. Both must reach a medical bay.",
    "Priority on the red survivor — get them out before starting on the orange one.",
    "Two casualties are down. Treat the red one first, then the orange.",
    "The red casualty is more urgent. Rescue them first; the orange one comes after.",
    "Locate both victims, but the red one must reach a bay before the orange one is moved.",
    "Mission: red casualty first. Once delivered, then save the orange one.",
    "Bring the red person to safety before touching the orange one.",
    "Order matters. Red casualty before orange. Both eventually go to bays.",
    "The red one is the priority. Don't load the orange casualty until the red one is delivered.",
    "Red comes first. Orange second. Two casualties total, both to the bays.",
    "First the red survivor reaches the green bay. Then go for the orange survivor.",
    "Save the red casualty first. The orange one must wait until the red is in a bay.",
    "Strict order: red before orange. Both end up in the medical bays.",
    "The red victim takes precedence. Deliver them, then handle the orange one.",
    "Don't start the orange evacuation until the red casualty is delivered.",
    "Get the red casualty to a bay first. Then evacuate the orange one.",
    "Red is the priority. Orange comes second. Both must reach the medical bays.",
    "Two people need help; the red one is more critical. Save them first.",
    "Treat the red casualty before the orange one. Both ultimately go to the bays.",
    "Order: red, then orange. Both must be delivered.",
    "Rescue the red one first. The orange survivor waits.",
]

SUMMARIES_CONSTRAINED_B_FIRST: List[str] = [
    "Rescue the orange casualty first; only after they are at the blue bay should you go for the red one.",
    "Priority is the orange casualty. Save them first. Then deal with the red one.",
    "There are two injured people. Do the orange one before the red one.",
    "Save the orange casualty before the red casualty.",
    "Orange victim first, then red. Both must reach a medical bay.",
    "Priority on the orange survivor — get them out before starting on the red one.",
    "Two casualties are down. Treat the orange one first, then the red.",
    "The orange casualty is more urgent. Rescue them first; the red one comes after.",
    "Locate both victims, but the orange one must reach a bay before the red one is moved.",
    "Mission: orange casualty first. Once delivered, then save the red one.",
    "Bring the orange person to safety before touching the red one.",
    "Order matters. Orange casualty before red. Both eventually go to bays.",
    "The orange one is the priority. Don't load the red casualty until the orange one is delivered.",
    "Orange comes first. Red second. Two casualties total, both to the bays.",
    "First the orange survivor reaches the blue bay. Then go for the red survivor.",
    "Save the orange casualty first. The red one must wait until the orange is in a bay.",
    "Strict order: orange before red. Both end up in the medical bays.",
    "The orange victim takes precedence. Deliver them, then handle the red one.",
    "Don't start the red evacuation until the orange casualty is delivered.",
    "Get the orange casualty to a bay first. Then evacuate the red one.",
    "Orange is the priority. Red comes second. Both must reach the medical bays.",
    "Two people need help; the orange one is more critical. Save them first.",
    "Treat the orange casualty before the red one. Both ultimately go to the bays.",
    "Order: orange, then red. Both must be delivered.",
    "Rescue the orange one first. The red survivor waits.",
]

SUMMARIES_HEAVY_STRICT_AB: List[str] = [
    "The red casualty is heavy and critical — focus the entire team on them first. Only after they reach the green bay should you start searching for the orange one.",
    "Two casualties, but the red one is in critical condition and must be evacuated first. Do not search for the orange one until the red is delivered.",
    "Red is heavy. The whole team must commit to red first. Once delivered, then locate and evacuate orange.",
    "Save the red casualty first; do not even look for the orange one until red is at the bay.",
    "The red one is critical. Get them out before scouting for the orange casualty.",
    "Red is in worse shape. Rescue them first. Then start searching for the orange one.",
    "Strict sequential rescue: deliver the red casualty, then begin the orange one.",
    "The red casualty needs all hands. Once delivered, the team can switch to orange.",
    "First mission: red casualty to the green bay. Second mission: locate and rescue orange.",
    "Red is the urgent one. Do not divide attention — finish red, then start orange.",
    "The team's full focus is on the red casualty. After the green bay, then move on to orange.",
    "Heavy casualty: red. Whole team. Then orange.",
    "Save red first, in full force. Orange waits until red is delivered.",
    "Red victim must reach the green bay before any search for the orange begins.",
    "Critical casualty is red. Evacuate first. Then begin orange phase.",
    "Treat the red casualty as the only mission until they are in the green bay. Then orange.",
    "Whole team rescues red first. Orange phase begins only after delivery.",
    "Sequential operation: red, then orange. Do not parallelise.",
    "Red is the heavy patient. Move them, deliver them, then begin orange.",
    "First red, then orange — and not concurrently. Red is critical.",
]

SUMMARIES_BY_AUTOMATON: Dict[str, List[str]] = {
    "free_order":          SUMMARIES_FREE_ORDER,
    "constrained_a_first": SUMMARIES_CONSTRAINED_A_FIRST,
    "constrained_b_first": SUMMARIES_CONSTRAINED_B_FIRST,
    "heavy_strict_ab":     SUMMARIES_HEAVY_STRICT_AB,
}

# --------------------------------------------------------------------------- #
#  Per-(automaton, joint_state) sub-task responses
#
#  Each entry is a short utterance describing what the team should be doing
#  given the current joint state under that automaton. We list 6–10 phrasings
#  per state — combined with random sampling, the per-trace responses are
#  diverse without us hand-writing 500 unique sentences per state.
# --------------------------------------------------------------------------- #

def _common_responses_default() -> Dict[Tuple[str, str], List[str]]:
    """Joint-state responses shared across automata where ordering doesn't apply."""
    return {
        ("unfound", "unfound"): [
            "Spread out and search for both casualties.",
            "Scouts: scan the area for the two victims.",
            "Both casualties are missing — fan out and locate them.",
            "Search phase. Locate both injured people.",
            "Both targets unaccounted for. Begin search.",
            "Cover the warehouse, find the two casualties.",
            "Discover both victims. Carriers stand by near the stretcher.",
            "Initial search: locate both casualties.",
        ],
        ("found", "unfound"): [
            "Red casualty located. Keep searching for the orange one.",
            "First casualty found. Carriers approach the stretcher; scouts continue searching.",
            "Red victim spotted. Scouts continue scanning for the orange one.",
            "One casualty down: red. Continue searching for the second.",
            "Casualty A acquired. Carriers move to pick up; scouts continue search.",
            "Red survivor found. Begin transport prep while scouts find the orange one.",
            "Red located. Orange still missing — keep searching.",
        ],
        ("unfound", "found"): [
            "Orange casualty located. Keep searching for the red one.",
            "Orange victim spotted. Scouts continue scanning for the red one.",
            "One casualty down: orange. Continue searching for the second.",
            "Casualty B acquired. Carriers move to pick up; scouts continue search.",
            "Orange survivor found. Begin transport prep while scouts find the red one.",
            "Orange located. Red still missing — keep searching.",
            "First casualty found. Carriers approach the stretcher; scouts continue searching.",
        ],
        ("found", "found"): [
            "Both casualties located. Carriers, dock with the stretcher and pick one up.",
            "Both victims found. Begin transport.",
            "Search complete. Pick a casualty and carry them to a bay.",
            "Both targets acquired. Carriers prepare to load and transport.",
            "Both injured people located. Start moving them to the bays.",
            "Search done. Carriers, sync at the stretcher and pick up a casualty.",
            "Locations confirmed. Begin transport — choose either casualty.",
        ],
        ("loaded", "unfound"): [
            "Carrying the red casualty. Deliver to the green bay; scouts keep looking for the orange one.",
            "Red on the stretcher, en route to the green bay. Scouts continue searching for orange.",
            "Transport in progress: red to green. Scouts, find the orange one.",
            "Red being moved to the green bay. Continue scouting for the orange casualty.",
            "Red loaded. Take to the green bay while scouts search for the missing orange.",
        ],
        ("loaded", "found"): [
            "Red is on the stretcher heading to the green bay. Orange located and waiting.",
            "Transporting red to the green bay; orange already found and waiting for pickup.",
            "Red being delivered. Orange identified — will be next.",
            "Red en route to green bay. Orange spotted, waiting for the stretcher to return.",
        ],
        ("unfound", "loaded"): [
            "Carrying the orange casualty. Deliver to the blue bay; scouts keep looking for the red one.",
            "Orange on the stretcher, en route to the blue bay. Scouts continue searching for red.",
            "Transport in progress: orange to blue. Scouts, find the red one.",
            "Orange being moved to the blue bay. Continue scouting for the red casualty.",
            "Orange loaded. Take to the blue bay while scouts search for the missing red.",
        ],
        ("found", "loaded"): [
            "Orange is on the stretcher heading to the blue bay. Red located and waiting.",
            "Transporting orange to the blue bay; red already found and waiting for pickup.",
            "Orange being delivered. Red identified — will be next.",
            "Orange en route to blue bay. Red spotted, waiting for the stretcher to return.",
        ],
        ("delivered", "unfound"): [
            "Red is delivered. Now find the orange casualty.",
            "Red safe at the green bay. Begin search for the orange one.",
            "First mission complete: red delivered. Locate the orange casualty.",
            "Red rescued. Scouts, find the orange casualty.",
        ],
        ("delivered", "found"): [
            "Red delivered. Orange located — carriers, prepare to load.",
            "Red safe. Orange identified, ready for pickup.",
            "First done. Orange found and waiting; carriers move to the stretcher.",
        ],
        ("delivered", "loaded"): [
            "Red delivered. Orange now on the stretcher heading for the blue bay.",
            "Red safe. Orange en route to blue.",
            "First mission done. Orange in transport.",
        ],
        ("unfound", "delivered"): [
            "Orange is delivered. Now find the red casualty.",
            "Orange safe at the blue bay. Begin search for the red one.",
            "First mission complete: orange delivered. Locate the red casualty.",
            "Orange rescued. Scouts, find the red casualty.",
        ],
        ("found", "delivered"): [
            "Orange delivered. Red located — carriers, prepare to load.",
            "Orange safe. Red identified, ready for pickup.",
            "First done. Red found and waiting; carriers move to the stretcher.",
        ],
        ("loaded", "delivered"): [
            "Orange delivered. Red now on the stretcher heading for the green bay.",
            "Orange safe. Red en route to green.",
            "First mission done. Red in transport.",
        ],
        ("delivered", "delivered"): [
            "Both casualties delivered. Mission complete.",
            "Mission accomplished. Both victims are at the bays.",
            "All casualties rescued. Stand down.",
            "Done. Both casualties safe.",
        ],
    }


# Constrained variants override messaging on relevant states to convey priority.
def _responses_constrained_a_first() -> Dict[Tuple[str, str], List[str]]:
    base = _common_responses_default()
    base[("found", "found")] = [
        "Both located. Carriers, pick up the red one first — orange waits.",
        "Both found. Red is the priority — load red first.",
        "Search complete. Red goes first; orange remains until red is delivered.",
        "Both casualties found. Begin transport with the red one.",
        "Red is the priority casualty. Load red, then orange after delivery.",
    ]
    base[("unfound", "found")] = [
        "Orange located, but the red one must go first. Continue searching for red.",
        "Orange spotted; red still missing and is the priority. Keep searching for red.",
        "Red is priority and not yet found. Orange waits. Search for red.",
        "Orange found but cannot be touched yet — red must go first.",
    ]
    base[("delivered", "unfound")] = [
        "Red delivered. Now go find the orange casualty.",
        "Priority complete. Red is safe — locate the orange casualty.",
        "Red mission done. Begin orange phase: locate them.",
    ]
    return base


def _responses_constrained_b_first() -> Dict[Tuple[str, str], List[str]]:
    base = _common_responses_default()
    base[("found", "found")] = [
        "Both located. Carriers, pick up the orange one first — red waits.",
        "Both found. Orange is the priority — load orange first.",
        "Search complete. Orange goes first; red remains until orange is delivered.",
        "Both casualties found. Begin transport with the orange one.",
        "Orange is the priority casualty. Load orange, then red after delivery.",
    ]
    base[("found", "unfound")] = [
        "Red located, but the orange one must go first. Continue searching for orange.",
        "Red spotted; orange still missing and is the priority. Keep searching for orange.",
        "Orange is priority and not yet found. Red waits. Search for orange.",
        "Red found but cannot be touched yet — orange must go first.",
    ]
    base[("unfound", "delivered")] = [
        "Orange delivered. Now go find the red casualty.",
        "Priority complete. Orange is safe — locate the red casualty.",
        "Orange mission done. Begin red phase: locate them.",
    ]
    return base


def _responses_heavy_strict_ab() -> Dict[Tuple[str, str], List[str]]:
    base = _common_responses_default()
    base[("unfound", "unfound")] = [
        "Whole team focuses on locating the red casualty — they are critical.",
        "Search the area for the red casualty. The orange one will come later.",
        "Find the red casualty first; orange is not in scope yet.",
        "Single objective right now: find the red survivor.",
        "Red is the heavy casualty. Locate them immediately. Ignore the orange one for now.",
    ]
    base[("found", "unfound")] = [
        "Red located. Whole team converges to load and deliver. Ignore orange for now.",
        "Red found — full attention on extracting them. Orange comes later.",
        "Red victim spotted. All carriers to the stretcher; scouts assist with cover.",
        "Red is critical and located. Move them now — orange phase comes after.",
    ]
    base[("loaded", "unfound")] = [
        "Carrying red to the green bay. Stay on this casualty — do not divert to orange.",
        "Red on the stretcher. Deliver before considering the orange casualty.",
        "Red in transport. Orange phase has not begun yet.",
        "Red en route to green bay. Full team escort; orange remains out of scope.",
    ]
    base[("delivered", "unfound")] = [
        "Red delivered. Orange phase begins now — locate the orange casualty.",
        "Red is safe at the green bay. Begin searching for the orange one.",
        "Critical mission done. Now locate the orange survivor.",
        "Red rescued. Switch the search to the orange casualty.",
    ]
    base[("delivered", "found")] = [
        "Red delivered. Orange located — carriers, prepare to load.",
        "Red safe. Orange identified, ready for pickup.",
        "Begin orange transport. Red mission complete.",
    ]
    base[("delivered", "loaded")] = [
        "Red delivered. Orange now in transport to the blue bay.",
        "Red safe. Orange en route.",
        "Final phase: orange delivery in progress.",
    ]
    return base


RESPONSES_BY_AUTOMATON: Dict[str, Dict[Tuple[str, str], List[str]]] = {
    "free_order":          _common_responses_default(),
    "constrained_a_first": _responses_constrained_a_first(),
    "constrained_b_first": _responses_constrained_b_first(),
    "heavy_strict_ab":     _responses_heavy_strict_ab(),
}


# --------------------------------------------------------------------------- #
#  Generation
# --------------------------------------------------------------------------- #

def _pick_response(rng: random.Random, automaton: str, joint_state: Tuple[str, str]) -> str:
    pool = RESPONSES_BY_AUTOMATON[automaton].get(joint_state)
    if not pool:
        # Fallback — covers any joint state we forgot to template (defensive).
        return f"State: A={joint_state[0]}, B={joint_state[1]}."
    return rng.choice(pool)


def _pick_summary(rng: random.Random, automaton: str) -> str:
    return rng.choice(SUMMARIES_BY_AUTOMATON[automaton])


def _augment_traces(automaton: str, raw_traces: List[dict], embedder, rng: random.Random) -> List[dict]:
    """Per-trace augmentation: pick a summary, pick per-step responses, embed all."""
    augmented: List[dict] = []

    # Embed all candidate summaries + responses for this automaton in one batch
    # for efficiency.
    summary_pool = SUMMARIES_BY_AUTOMATON[automaton]
    response_pool_by_state = RESPONSES_BY_AUTOMATON[automaton]
    flat_responses: List[str] = []
    response_index: Dict[Tuple[str, str], slice] = {}
    cursor = 0
    for state_key, pool in response_pool_by_state.items():
        flat_responses.extend(pool)
        response_index[state_key] = slice(cursor, cursor + len(pool))
        cursor += len(pool)

    print(f"  encoding {len(summary_pool)} summaries + {len(flat_responses)} response variants ...")
    summary_embs = embedder.encode(summary_pool, normalize_embeddings=False, convert_to_numpy=True)
    response_embs = embedder.encode(flat_responses, normalize_embeddings=False, convert_to_numpy=True)
    assert summary_embs.shape[1] == EMBEDDING_DIM, summary_embs.shape
    assert response_embs.shape[1] == EMBEDDING_DIM, response_embs.shape

    for trace in raw_traces:
        summary_idx = rng.randint(0, len(summary_pool) - 1)
        summary = summary_pool[summary_idx]
        y = summary_embs[summary_idx].tolist()

        # Sample ONE response per unique state per trace, then reuse — matches
        # the four_flags pipeline (consistent textual interpretation of a
        # state within a single trace; diversity comes from across-trace
        # sampling).
        unique_keys = {(p[0], p[1]) for p in trace["states"]}
        per_state_choice: Dict[Tuple[str, str], Tuple[str, List[float]]] = {}
        for key in unique_keys:
            pool_slice = response_index.get(key)
            if pool_slice is None:
                per_state_choice[key] = (f"State: A={key[0]}, B={key[1]}.", [0.0] * EMBEDDING_DIM)
                continue
            local_idx = rng.randint(0, pool_slice.stop - pool_slice.start - 1)
            global_idx = pool_slice.start + local_idx
            per_state_choice[key] = (
                flat_responses[global_idx],
                response_embs[global_idx].tolist(),
            )
        responses_per_step: List[str] = []
        h_per_step: List[List[float]] = []
        for state_pair in trace["states"]:
            text, emb = per_state_choice[(state_pair[0], state_pair[1])]
            responses_per_step.append(text)
            h_per_step.append(emb)

        # subtask_decoder_labels: training pipeline expects a per-step vector;
        # we emit zeros until / unless the EventRNN is configured with
        # ``use_label_decoder=True``. STATE_DIM-zeros is a safe placeholder.
        T = len(trace["states"])
        subtask_decoder_labels = [[0] * SUBTASK_DECODER_DIM for _ in range(T)]

        augmented.append({
            **trace,
            "summary": summary,
            "y": y,
            "responses": responses_per_step,
            "h": h_per_step,
            "subtask_decoder_labels": subtask_decoder_labels,
        })

    return augmented


def main() -> None:
    rng = random.Random(RNG_SEED)

    print(f"Loading embedding model {EMBEDDING_MODEL!r} (1024-dim) on CPU ...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    all_augmented: List[dict] = []
    for automaton, fname in DFA_FILES:
        path = WALKS_DIR / fname
        print(f"\n[{automaton}] loading {path} ...")
        with path.open() as f:
            raw_traces = json.load(f)
        print(f"[{automaton}] {len(raw_traces)} traces")
        augmented = _augment_traces(automaton, raw_traces, embedder, rng)
        all_augmented.extend(augmented)
        print(f"[{automaton}] augmented {len(augmented)} traces")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(all_augmented, f)
    print(f"\nWrote {len(all_augmented)} augmented traces ➜ {OUT_PATH}")


if __name__ == "__main__":
    main()
