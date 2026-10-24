# NFL Analytics

Personal learning project. The point is to learn and experiment with models on NFL
play-by-play data (1999+), not to ship anything. Explaining *why* something works or
fails is as valuable as the fix.

Long-term goal: play-calling models (scenario → play) that go head to head inside a
game simulator. The simulator is built from learned outcome models; penalties and pass
outcomes exist so far. Plays are currently treated as independent. Sequence models
(LSTM/transformer over a game) that can judge whether a strategy is working are a
planned later step.

## Layout
- `core/` — generic, NFL-agnostic training infrastructure. Keep it free of NFL imports.
  - `trainer/pipeline.py` `PipelineDriver`: Optuna hparam search with K-fold CV, then trains an ensemble.
  - `trainer/solver.py` `Solver`: single-model train loop; infers classification vs regression from the criterion.
  - `trainer/ensemble.py` `EnsembleModel`: holds N models, returns (mean, std); owns save/load.
  - `utils/lib/data_handling.py`: train/test split to tensors, class weights.
- `nfl/` — domain code.
  - `data_management/data_manager.py` `DataManager`: load CSVs, clean, map to enums, one-hot encode. `config.yaml` lists column groups.
  - `lib/constants.py` (feature column lists like `SCENARIO_COLS`, `PLAY_COLS`), `lib/nfl_enums.py`, `lib/datamodels.py`.
  - `models/`: `WPAPredictor`, `PassOutcomeEngine`, `PassDistanceEngine` (nn.Modules).
  - `train_*_predictor.py`: entry points that wire data → `PipelineDriver` → saved ensemble.
  - `simulators/`: `PlaySimulator`, `Driver` that consume trained models.
  - `data/` (CSVs, gitignored) and `artifacts/` (`.pth`, gitignored).
- `main.ipynb` — the user's scratchpad that loads or trains each model. Don't edit it unless asked.

## Commands
Run everything from the repo root; data paths are built from `Path.cwd()`.
- Env: `.venv/bin/python` (`make setup` to create it)
- `make format` (ruff format), `make lint` (ruff check --fix, mypy), `make test` (pytest tests/)
- Ruff: line length 88, double quotes, py310 target.

## Current state / known issues
- `tests/` is empty on purpose for now; tests are low priority. Don't add them unless asked.
- Open problem: WPA predictor CV loss during hparam selection doesn't carry over to
  final ensemble training (generalization gap). Pass predictor is "ok"; class-weighted
  loss made it worse.
- Measured (2021–2025 seasons, 2026-10-23): consecutive-play WPA correlation is about
  -0.02, and game_id explains under 1% of WPA variance. Within-game leakage of the
  *target* is therefore negligible. Game-state features (wp lag-1 corr about 0.75) are
  correlated within a game, so all splits are grouped by `game_id` (pass `group_col` /
  `groups` through `create_train_test_split` and `PipelineDriver`). `game_id` survives
  the substring-based `drop_columns` only because it is listed in `keep_columns`.

## Working conventions
- Training runs can be long. Prefer small `num_epoch` / `n_trials` smoke runs to check
  a change works before suggesting a full run, and say which you ran.
- Report metrics with the split they came from (CV fold, held-out test, train).
- Don't commit `.pth` files or data. Commit only when asked.
