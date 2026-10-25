import numpy as np
import pandas as pd


def class_rates(
    probs: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean predicted probability vs actual frequency of each class, in percent.

    For a model whose output is sampled from, these should match overall and within
    any slice of the inputs (pass groups to get one row per group value).
    """
    n_classes = probs.shape[1]
    probs = probs.astype(np.float64)  # float32 prints as 56.330002 even when rounded
    groups = np.zeros(len(y)) if groups is None else np.asarray(groups)
    rows = {}
    for g in np.unique(groups):
        mask = groups == g
        actual = np.bincount(y[mask], minlength=n_classes) / mask.sum()
        predicted = probs[mask].mean(axis=0)
        rows[g] = {"n": int(mask.sum())}
        for i, name in enumerate(class_names):
            rows[g][f"{name} pred"] = predicted[i] * 100
            rows[g][f"{name} act"] = actual[i] * 100
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def reliability(prob: np.ndarray, hit: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Rows bucketed by one class's predicted probability (quantile bins), with each
    bucket's mean predicted probability vs actual rate, in percent.

    class_rates can match on average while the model is overconfident (spreads
    predictions too wide) or underconfident (squeezes them toward the average);
    this shows which.
    """
    prob = prob.astype(np.float64)
    buckets = pd.qcut(prob, n_bins, duplicates="drop")
    df = pd.DataFrame({"predicted": prob * 100, "actual": hit * 100.0})
    return (
        df.groupby(buckets, observed=True)
        .agg(
            n=("actual", "size"),
            predicted=("predicted", "mean"),
            actual=("actual", "mean"),
        )
        .round(2)
    )
