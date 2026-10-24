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
    Model-agnostic: any `model_cls(input_size=..., **model_hparams)` works (MLP now; LSTM/transformer
    later). `batch_size`/`lr`/`weight_decay` (`TRAINING_KEYS`) go to training, everything else to the
    model. `_make_solver` is the only place models/optimizers are built; the optimizer is the
    `optimizer_cls` arg (default AdamW) so search and final training always match.
    Param space values: `(low, high, dist)` tuples are searched, plain values are fixed.
  - `trainer/solver.py` `Solver`: single-model train loop; infers classification vs regression from the criterion.
  - `models/mlp.py` `MLP`: generic feed-forward net; NFL models subclass it.
  - `trainer/ensemble.py` `EnsembleModel`: holds N models, returns (mean, std); owns save/load.
  - `utils/lib/data_handling.py`: train/test split to tensors, class weights.
- `nfl/` — domain code.
  - `data_management/data_manager.py` `DataManager`: load CSVs, clean, map to enums, one-hot encode. `config.yaml` lists column groups.
  - `lib/constants.py` (feature column lists like `SCENARIO_COLS`, `PLAY_COLS`), `lib/nfl_enums.py`, `lib/datamodels.py`.
  - `models/`: `WPAPredictor`, `PassOutcomeEngine`, `PassDistanceEngine`, `PreSnapPenaltyEngine`/`LiveBallPenaltyEngine` (thin `MLP` subclasses that fix output size/activation).
  - `train_*_predictor.py`: entry points that wire data → `PipelineDriver` → saved ensemble.
  - `simulators/`: `PlaySimulator`, `Driver` that consume trained models.
    `play_caller.py`: `PlayCall` (pre-snap choices), `all_play_calls()` (55 valid calls),
    `WPAPlayCaller` (scores every call in one batch, picks max of `mean - k*std`). It builds
    features by running raw rows through the same `DataManager` code as training and reads
    the column order from the ensemble's saved `feature_names`.
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
  loss made it worse. Until 2026-10-23 the search used Adam and the final ensemble AdamW,
  so the tuned `weight_decay` meant different things in each; may explain part of the gap.
- Measured (2021–2025 seasons, 2026-10-23): consecutive-play WPA correlation is about
  -0.02, and game_id explains under 1% of WPA variance. Within-game leakage of the
  *target* is therefore negligible. Game-state features (wp lag-1 corr about 0.75) are
  correlated within a game, so all splits are grouped by `game_id` (pass `group_col` /
  `groups` through `create_train_test_split` and `PipelineDriver`). `game_id` survives
  the substring-based `drop_columns` only because it is listed in `keep_columns`.
- Feature scaling lives in `PipelineDriver` (`scale_features=True`) and the fitted
  mean/std are stored as `EnsembleModel` buffers, so callers always pass raw features.
  Preliminary check (2023–25, 15 epochs): unscaled models collapsed to predicting a
  constant (test MSE equal to the predict-0 baseline of 0.00207); scaled reached 0.00196
  (about 5% better). tanh made no difference once inputs were scaled. Per-play WPA is
  mostly outcome noise, so expect a low R²; the model's job is the expected WPA.

- Categorical play columns (`play_type`, `pass_location`, `run_location`) are one-hot
  encoded against fixed categories (`CATEGORICAL_PLAY_COLS`), so feature columns don't
  depend on the data. Checkpoints from before 2026-10-24 use the old 27 features and
  have no `feature_names`, so they can't drive `WPAPlayCaller`; retrain.
- Play-caller findings (2026-10-24, quick model, 2023–25):
  - `air_yards`, `pass_length` and `pass_location` are present on every thrown ball but
    missing on every sack, so for a pass they encoded "was not sacked". On 1st & 10 own
    20–30, thrown passes average +0.0058 WPA vs +0.0033 for all dropbacks. Scrambles are
    all dropbacks labelled `run`. Enumerated passes were overrated.
  - Fixed in `DataManager._describe_dropbacks_as_pass_calls` (part of `clean_data`):
    play columns describe the *call*. Every dropback is a pass (scrambles lose their run
    direction); sacks/scrambles get depth+direction copied from a random thrown pass
    (seeded). `qb_scramble` is no longer a feature. Known tradeoff: sack risk is spread
    evenly over short and deep calls, since intended depth isn't recorded on sacks.
  - Where data exists the model is accurate (4th & 15 own 20: punt predicted +0.0156,
    actual +0.0161). Where coaches almost never call a play (under 20 passes on 4th & 10+
    in own territory in 10 seasons) it extrapolates (+0.034), and ensemble std doesn't
    flag it. A behaviour model P(call | scenario) is the planned filter.
  - `air_yards` was removed from `PLAY_COLS` (2026-10-24). A coach calls short/deep,
    not an exact depth; training on realized depth while the caller fed one typical
    depth per call misstated completion rates and narrowed the yards spread. Older
    checkpoints list it in `feature_names`; retrain.

- Pass models (2026-10-24): `DataManager.select_pass_plays` keeps dropbacks on a real
  down with no penalty (the simulator applies penalties separately; two-point tries are
  the only pass/run plays with no down). `PassOutcome.SCRAMBLE` exists because
  scrambles used to fall through to INCOMPLETE. `PassDistanceEngine` outputs logits
  over yard bins `PASS_YARDS_RANGE` (-30..99) for complete/scramble/sack, with the
  outcome one-hot as input; `yards_proba`/`sample_yards` mask gains past the goal line
  and losses past the end line. Judge both models by cross-entropy against the
  class-frequency baseline (and CRPS for yards), not accuracy: the simulator samples
  probabilities. Old pass-outcome accuracy (~64%) predates the sack-leak fix.
- Penalty engines (2026-10-24, `train_penalty_predictor.py`, replaced the never-trained
  per-type LightGBM `PenaltyPredictor`). The data records at most one penalty per play
  (2+ flags on 0.35% of penalty plays, offsetting 0.03%), so each engine is one softmax
  over [none, listed (type, side) classes in `PRE_SNAP_PENALTIES` /
  `LIVE_BALL_PENALTIES`, other offense, other defense]. Rare types in the other classes
  are named by frequencies saved in the ensemble's `config["other_penalties"]`;
  `sample_flags` returns `Flag | None` per row. Two engines because pre-snap fouls have
  no call recorded: one model would learn "call missing = false start".
  Training takes `get_data` output (clean_data drops the no_play rows that hold most
  fouls). `DataManager._describe_nullified_calls` (in `get_data`, before desc/posteam/
  penalty_team are dropped) adds `penalty_on_offense` and parses the call from desc on
  wiped-out plays; `create_penalty_targets` builds the labels. 2023-25: pre-snap foul
  on 2.8% of snaps, live-ball foul on 5.5% of snapped plays; DPI 0.75% of short vs 3.6%
  of deep passes. Enforcement (which yards, replay down, accept/decline) isn't built.

## Working conventions
- Training runs can be long. Prefer small `num_epoch` / `n_trials` smoke runs to check
  a change works before suggesting a full run, and say which you ran.
- Report metrics with the split they came from (CV fold, held-out test, train).
- Don't commit `.pth` files or data. Commit only when asked.
