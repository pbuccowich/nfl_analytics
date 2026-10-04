from typing import Protocol

import torch
import torch.nn as nn

from core.trainer.ensemble import EnsembleModel
from nfl.lib.datamodels import Scenario


class NNPolicyModel(Protocol):
    model: EnsembleModel | nn.Module
    def __init__(self, model: EnsembleModel | nn.Module):
        self.model = model

    def get_possible_plays(self, scenario: Scenario) -> list[torch.Tensor]:
        return []

    def select_play(self, scenario: Scenario) -> torch.Tensor:
        """Evaluates scenario context and selects the optimal play."""
        possible_plays = self.get_possible_plays(scenario)
        if not possible_plays:
            raise ValueError("No possible plays returned for scenario.")

        scenario_tensor = scenario.to_tensor()

        play_performances = []
        for play in possible_plays:
            # Concatenate scenario features with play features along feature dimension
            input_tensor = torch.cat([scenario_tensor, play], dim=0).unsqueeze(0)

            with torch.no_grad():
                score = self.model(input_tensor).item()

            play_performances.append(score)

        best_idx = max(range(len(play_performances)), key=lambda i: play_performances[i])
        return possible_plays[best_idx]

class Driver:
    def __init__(self, policy_model: NNPolicyModel):
        self.policy = policy_model

    def _generate_starting_scenario(self) -> Scenario:
        # Placeholder for generating initial scenario
        pass

    def run_drive(self, current_scenario: Scenario | None = None) -> tuple[bool, int, Scenario | None]:
        drive_in_progress: bool = True
        if current_scenario is None:
            current_scenario = self._generate_starting_scenario()

        game_over: bool = False
        points_scored: int = 0

        while drive_in_progress:
            play = self.policy.select_play(current_scenario)
            current_scenario, transition_result = self.simulator.get_play_outcome(current_scenario, play)
            if transition_result.is_game_over or transition_result.is_touchdown or transition_result.is_safety or transition_result.is_turnover:
                drive_in_progress = False
                if transition_result.is_touchdown:
                    points_scored = 7
                elif transition_result.is_safety:
                    points_scored = -2
                if transition_result.is_game_over:
                    game_over = True

        return game_over, points_scored, current_scenario

