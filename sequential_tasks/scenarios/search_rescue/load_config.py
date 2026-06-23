"""Hydra/kwargs config loader for the Search-and-Rescue scenario.

Mirrors the declarative pattern in ``scenarios/four_flags/load_config.py``.
"""
from collections.abc import Mapping


PARAMS = [
    # --- Arena -------------------------------------------------------------
    ("x_semidim", 4.0), ("y_semidim", 3.0),
    ("shared_reward", False),

    # --- Team composition --------------------------------------------------
    ("n_scouts", 2), ("n_carriers", 2),

    # --- Geometry ----------------------------------------------------------
    ("agent_radius", 0.17),
    ("casualty_radius", 0.20),
    ("stretcher_radius", 0.30),
    ("bay_radius", 0.40),
    ("min_casualty_separation", 1.0),

    # --- Detection / docking thresholds -----------------------------------
    # dock_radius widened (0.40 → 0.55) so the two carriers can stay synced
    # while moving the stretcher together — at 0.40 they typically lose
    # sync after a few steps and the stretcher stops following.
    ("detect_radius", 0.45),
    ("dock_radius", 0.55),
    ("pickup_radius", 0.55),

    # --- Rewards ----------------------------------------------------------
    # Re-tuned after second run plateaued at discovery-only (per-step shaping
    # was -0.02/step at dist=2, dwarfed by +20 discovery bonus → no gradient
    # toward loading sub-task). 3× stronger; still well below the +30/+60
    # loading and delivery bonuses to keep correct task ordering.
    ("shaping_factor_carrier", 30.0),
    ("shaping_factor_scout", 15.0),
    # r_discover halved (20 → 10) — discovery reward was so high it leaked
    # to carriers via shared policy params, making them imitate the scout
    # "go to casualty" behavior. Smaller discovery still rewards scouts but
    # doesn't dominate the policy gradient.
    ("r_discover", 10.0),
    ("r_load", 30.0),
    ("r_deliver", 60.0),
    ("r_sync_step", 0.05),
    # r_terminal bumped 100 → 300 to dominate "stay at the bay" attraction
    # after first delivery. Previous run plateaued at 1-of-2 because the
    # team had no strong incentive to leave bay_B and find casualty A.
    ("r_terminal", 300.0),
    ("collision_penalty", 5.0),

    # --- Dynamics ----------------------------------------------------------
    # All agents share the same action-space bounds (BenchMARL groups them
    # under a single "agents" group; per-agent bounds break the spec API).
    # Role-conditioned speed differences live in the policy via role_one_hot.
    ("use_velocity_controller", True),
    ("agent_weight", 1.0),
    ("agent_v_range", 1.0),
    ("agent_a_range", 1.0),
    ("min_collision_distance", 0.1),
    ("linear_friction", 0.1),

    # --- Communication / GNN ---------------------------------------------
    ("use_gnn", False), ("comm_dim", 1),
    ("comms_radius", 1.5),

    # --- Language ---------------------------------------------------------
    # event_dim/state_dim must match sequence_models/search_rescue/random_automaton_walk_search_rescue.py
    ("embedding_size", 1024), ("use_embedding_ratio", 1.0),
    ("event_dim", 7), ("state_dim", 10), ("num_automata_bits", 2),
    ("load_dataset", True),

    # --- Paths ------------------------------------------------------------
    # Path is relative to the project root (parents[2] from this file). The
    # dataset is at sequence_models/data/... per build_augmented_dataset_search_rescue.py.
    # Earlier configs pointed to "data/search_rescue/..." which didn't exist —
    # the silent FileNotFoundError catch meant training ran with all-zero task
    # embeddings (huge bug; only caught when adding variant gating in 2026-05).
    ("data_json_path", "sequence_models/data/search_rescue/dataset_search_rescue_ALL.json"),
    ("sequence_model_path", "sequence_models/search_rescue/search_rescue.pth"),
    ("slm_model_path", "sequence_models/search_rescue/distilled_slm.pth"),  # for SLM scenario only
    ("latmos_model_path", "sequence_models/search_rescue/latmos.pth"),       # for LATMOS scenario only
    ("tiny_rnn_model_path", "sequence_models/search_rescue/tiny_rnn.pth"),   # for tiny-RNN scenario only
    ("tiny_rnn_latent_dim", 128),
    ("tiny_rnn_input_dim", 32),
    ("policy_config_path", "../../conf"),
    ("policy_config_name", "config.yaml"),
    ("policy_restore_path", "checkpoints/search_rescue/policy.pt"),
]

PARAMS = [
    (dest, key, default)
    for entry in PARAMS
    for dest, key, default in [(
        entry if isinstance(entry, tuple) and len(entry) == 3 else
        (entry[0], entry[0], entry[1]) if isinstance(entry, tuple) else
        (entry, entry, None)
    )]
]


def load_scenario_config(source, env):
    is_mapping = isinstance(source, Mapping)
    for dest_attr, key, default in PARAMS:
        if is_mapping:
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        setattr(env, dest_attr, value)

    env.n_agents = env.n_scouts + env.n_carriers
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = (
        env.agent_v_range if env.use_velocity_controller else env.agent_f_range
    )
    # Camera. VMAS computes cam_range = zoom * zoom when agents are within
    # the zoom radius, so the half-width of the view is zoom². To make the
    # view exactly span the arena (half-width = max(x_semidim, y_semidim)),
    # set zoom to sqrt of that. (Setting zoom=1 was too tight — a 2x2 view;
    # setting zoom=4 was too loose — a 32x32 view of an 8x6 arena.)
    import math as _math
    env.viewer_zoom = _math.sqrt(max(env.x_semidim, env.y_semidim))
