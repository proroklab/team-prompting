"""Held-out paraphrases for the paraphrase-robustness experiment (R17 §1, R9 §4).

The trained policy never saw these exact phrasings. They are semantically
identical to the training summaries (same DFA variant) but use deliberately
different wording, syntax, and lexicon. Embedding with the same gte-large
encoder used at training time should land them in a similar region of the
embedding space — if the policy is genuinely reading semantic content, it
will perform comparably on these inputs to its training distribution.
"""

HELDOUT_PARAPHRASES = {
    "free_order": [
        "Quickly — fetch both wounded individuals and bring them to the medical stations.",
        "Operation: extract both injured people from the warehouse. Either delivery order is acceptable.",
        "Two persons need help. Find them and transport them to safety. Pick whichever you prefer first.",
        "Mission: rescue two victims and bring each to a designated medical area. No required ordering.",
        "Two people require evacuation. The team must locate and move each to a medical bay.",
        "Find the two patients in the area and carry each one to a bay; the sequence is unimportant.",
        "Recover the two casualties and deliver them; there is no priority between the two.",
    ],
    "constrained_a_first": [
        "Strict priority: the red victim must reach the green bay before the orange one is touched.",
        "First priority is the red one — get them safely to their bay before dealing with orange.",
        "Order matters here: rescue red, deliver red, then turn to orange. Do not deviate.",
        "Red must be evacuated first. Touch the orange casualty only after red has been delivered.",
        "Phase 1: red to its bay. Phase 2: orange. Do not interleave the missions.",
        "The red individual has priority — finish that rescue completely before approaching orange.",
        "Sequential operation, red first. The orange victim is off-limits until red is delivered.",
    ],
    "constrained_b_first": [
        "Strict priority: the orange victim must reach the blue bay before the red one is touched.",
        "First priority is the orange one — get them safely to their bay before dealing with red.",
        "Order matters here: rescue orange, deliver orange, then turn to red. Do not deviate.",
        "Orange must be evacuated first. Touch the red casualty only after orange has been delivered.",
        "Phase 1: orange to its bay. Phase 2: red. Do not interleave the missions.",
        "The orange individual has priority — finish that rescue completely before approaching red.",
        "Sequential operation, orange first. The red victim is off-limits until orange is delivered.",
    ],
    "heavy_strict_ab": [
        "Red is critically injured — full team commits to red, no diversion. Only after red is at the bay does the search for orange begin.",
        "Operation phase 1: red to its bay with the whole team. Operation phase 2: locate and deliver orange. Strictly sequential, no overlap.",
        "Red is the heavy casualty. The entire team must focus on red first; do not let scouts wander toward orange until red is at the bay.",
        "The red victim must be your only concern until they are delivered. Orange is invisible until red is at the green bay.",
        "Full focus on red. Once red is safe, then — and only then — begin the orange rescue.",
        "Red is critical and must be delivered before the orange phase opens. Do not search for orange prematurely.",
    ],
}


def all_paraphrases_flat():
    """All paraphrases flat with their variant id. Used to embed in one batch."""
    out = []
    from sequence_models.search_rescue.random_automaton_walk_search_rescue import all_configs
    name_to_id = {c.name: c.automaton_id for c in all_configs()}
    for vname, paras in HELDOUT_PARAPHRASES.items():
        vid = name_to_id[vname]
        for p in paras:
            out.append((vid, vname, p))
    return out
