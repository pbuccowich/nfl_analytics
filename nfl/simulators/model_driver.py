from nfl.lib.datamodels import Scenario
from nfl.simulators.play_caller import WPAPlayCaller


class Driver:
    def __init__(self, policy_model: WPAPlayCaller):
        self.policy = policy_model

    def _generate_starting_scenario(self) -> Scenario:
        # Placeholder for generating initial scenario
        pass

    def run_drive(
        self, current_scenario: Scenario | None = None
    ) -> tuple[bool, int, Scenario | None]:
        drive_in_progress: bool = True
        if current_scenario is None:
            current_scenario = self._generate_starting_scenario()

        game_over: bool = False
        points_scored: int = 0

        while drive_in_progress:
            play = self.policy.select_play(current_scenario)
            current_scenario, transition_result = self.simulator.get_play_outcome(
                current_scenario, play
            )
            end_of_drive = any(
                [
                    transition_result.is_game_over,
                    transition_result.is_touchdown,
                    transition_result.is_safety,
                    transition_result.is_turnover,
                ]
            )
            if end_of_drive:
                drive_in_progress = False
                if transition_result.is_touchdown:
                    points_scored = 7
                elif transition_result.is_safety:
                    points_scored = -2
                if transition_result.is_game_over:
                    game_over = True

        return game_over, points_scored, current_scenario
