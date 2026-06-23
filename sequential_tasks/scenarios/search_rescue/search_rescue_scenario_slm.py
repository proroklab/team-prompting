"""Scenario subclass that swaps the EventRNN-based LanguageUnit for the
distilled-SLM-based one.

Inherits everything from ``SearchRescueScenario`` (RL subclass): same
atomic-prop detector, same reward, same observation contract, same agent
setup. Only ``_init_language_unit`` is overridden.

Configured via ``conf/search_rescue/task/search_rescue_slm_rl.yaml`` which
references ``scenario_class:
scenarios.search_rescue.search_rescue_scenario_slm.SearchRescueScenarioSLM``.
"""

from __future__ import annotations

from vmas.simulator.core import World

from scenarios.search_rescue.search_rescue_scenario_rl import SearchRescueScenario
from scenarios.search_rescue.language_slm import LanguageUnitSLM, load_slm_model


class SearchRescueScenarioSLM(SearchRescueScenario):
    """Drop-in scenario using the distilled SLM as the task / sub-task encoder."""

    def _init_language_unit(self, world: World) -> None:
        load_slm_model(
            model_path=getattr(self, "slm_model_path", "sequence_models/search_rescue/distilled_slm.pth"),
            device=world.device,
        )
        self.language_unit = LanguageUnitSLM(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
        if self.load_dataset:
            try:
                self.language_unit.load_sequence_data(json_path=self.data_json_path, device=world.device)
            except FileNotFoundError:
                pass
