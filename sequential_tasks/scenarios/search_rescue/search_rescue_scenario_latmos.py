"""LATMOS scenario: drop-in replacement for the EventRNN encoder.

Inherits everything from SearchRescueScenario (RL subclass) and overrides
only ``_init_language_unit`` to instantiate ``LanguageUnitLATMOS`` and load
the trained LATMOS weights instead of the EventRNN.
"""

from __future__ import annotations

from vmas.simulator.core import World

from scenarios.search_rescue.search_rescue_scenario_rl import SearchRescueScenario
from scenarios.search_rescue.language_latmos import LanguageUnitLATMOS, load_latmos_model


class SearchRescueScenarioLATMOS(SearchRescueScenario):
    def _init_language_unit(self, world: World) -> None:
        load_latmos_model(
            model_path=getattr(self, "latmos_model_path", "sequence_models/search_rescue/latmos.pth"),
            device=world.device,
        )
        self.language_unit = LanguageUnitLATMOS(
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
