"""Scenario subclass using the tiny RNN as the task / sub-task encoder.

Inherits the BenchMARL-ready ``SearchRescueScenario`` and overrides only
``_init_language_unit`` to swap in the tiny RNN + LanguageUnitTinyRNN.
"""

from __future__ import annotations

from vmas.simulator.core import World

from scenarios.search_rescue.search_rescue_scenario_rl import SearchRescueScenario
from scenarios.search_rescue.language_tiny_rnn import LanguageUnitTinyRNN, load_tiny_rnn_model
from sequence_models.search_rescue.distilled_rnn_tiny.model import (
    DEFAULT_LATENT_DIM, DEFAULT_INPUT_DIM,
)


class SearchRescueScenarioTinyRNN(SearchRescueScenario):
    def _init_language_unit(self, world: World) -> None:
        latent_dim = getattr(self, "tiny_rnn_latent_dim", DEFAULT_LATENT_DIM)
        input_dim = getattr(self, "tiny_rnn_input_dim", DEFAULT_INPUT_DIM)
        load_tiny_rnn_model(
            model_path=getattr(self, "tiny_rnn_model_path", "sequence_models/search_rescue/tiny_rnn.pth"),
            device=world.device,
            latent_dim=latent_dim,
            input_dim=input_dim,
        )
        self.language_unit = LanguageUnitTinyRNN(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            latent_dim=latent_dim,
            device=world.device,
        )
        # Camera anchored to world bounds (same as the base SLM/LATMOS path).
        self.visualize_semidims = True
        if self.load_dataset:
            try:
                self.language_unit.load_sequence_data(json_path=self.data_json_path, device=world.device)
            except FileNotFoundError:
                pass
