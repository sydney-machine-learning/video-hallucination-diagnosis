from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest" / "evaluation_manifest.csv"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RESPONSE_FILES = [
    ("Qwen2.5-Omni-3B", ROOT / "data" / "model_responses" / "qwen2_5_omni_3b_responses.csv"),
    ("GPT-5.4", ROOT / "data" / "model_responses" / "gpt_5_4_responses.csv"),
    ("GPT-5.5", ROOT / "data" / "model_responses" / "gpt_5_5_responses.csv"),
    ("GPT-5.6", ROOT / "data" / "model_responses" / "gpt_5_6_responses.csv"),
]

VALID_PARSE_STATUS = {"ok", "json", "parsed", "regex"}
BOOTSTRAP_REPEATS = 1000


def read_responses() -> pd.DataFrame:
    frames = []
    for display_name, path in RESPONSE_FILES:
        frame = pd.read_csv(path, keep_default_na=False)
        frame = frame.drop(
            columns=[
                "final_correct_answer",
                "final_A_language_prior",
                "final_A_vision",
                "final_uncertainty_label",
            ],
            errors="ignore",
        )
        frame["model_display"] = display_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_items() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    labels = manifest[
        [
            "question_id",
            "video_split",
            "clip_id",
            "vhd_category",
            "task",
            "stimulus_type",
            "final_correct_answer",
            "final_A_language_prior",
            "final_A_vision",
            "final_uncertainty_label",
        ]
    ]
    responses = read_responses()
    scored = responses.merge(labels, on="question_id", how="left", validate="many_to_one")
    scored["selected_choice"] = scored["selected_choice"].astype(str).str.upper().str.strip()
    scored["confidence_norm"] = pd.to_numeric(scored["confidence"], errors="coerce")
    scored["is_valid_response"] = (
        scored["parse_status"].astype(str).str.lower().isin(VALID_PARSE_STATUS)
        & scored["selected_choice"].isin(["A", "B"])
        & scored["confidence_norm"].between(0, 1)
    )
    scored["is_correct"] = np.where(
        scored["is_valid_response"],
        (scored["selected_choice"] == scored["final_correct_answer"].str.upper()).astype(float),
        np.nan,
    )
    scored["is_prior_choice"] = np.where(
        scored["is_valid_response"] & (scored["final_A_language_prior"] != ""),
        (scored["selected_choice"] == scored["final_A_language_prior"].str.upper()).astype(float),
        np.nan,
    )
    return scored


def expected_calibration_error(group: pd.DataFrame) -> float:
    group = group.dropna(subset=["confidence_norm", "is_correct"]).copy()
    if group.empty:
        return np.nan
    bins = np.linspace(0.0, 1.0, 6)
    group["bin"] = pd.cut(group["confidence_norm"], bins, include_lowest=True, labels=False)
    total = len(group)
    value = 0.0
    for _, bin_group in group.groupby("bin", observed=True):
        value += len(bin_group) / total * abs(
            bin_group["confidence_norm"].mean() - bin_group["is_correct"].mean()
        )
    return float(value)


def per_clip_scores(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, clip), group in scored.groupby(["model_display", "clip_id"]):
        accuracy = group[group["task"] == "accuracy"]["is_correct"].mean()
        pdi = group[group["task"] == "vhd_i_pdi"]["is_prior_choice"].mean()
        hce = expected_calibration_error(group[group["task"] == "vhd_ii_hce"])
        rows.append(
            {
                "model_display": model,
                "clip_id": clip,
                "accuracy": accuracy,
                "pdi": pdi,
                "hce": hce,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: pd.Series,
    label: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260711)
    samples = np.empty(repeats, dtype=float)
    for i in range(repeats):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
        if (i + 1) % 100 == 0 or i + 1 == repeats:
            print(f"{label}: bootstrap sample {i + 1}/{repeats}", flush=True)
    return float(values.mean()), *np.percentile(samples, [2.5, 97.5])


def bootstrap_hce_ci(
    group: pd.DataFrame,
    label: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float, float]:
    group = group.dropna(subset=["confidence_norm", "is_correct"]).copy()
    if group.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260711)
    clip_index, clip_ids = pd.factorize(group["clip_id"], sort=False)
    n_clips = len(clip_ids)
    if n_clips == 0:
        return np.nan, np.nan, np.nan
    confidence = group["confidence_norm"].to_numpy(dtype=float)
    correct = group["is_correct"].to_numpy(dtype=float)
    bin_index = pd.cut(
        confidence,
        np.linspace(0.0, 1.0, 6),
        include_lowest=True,
        labels=False,
    )
    samples = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sampled = rng.integers(0, n_clips, size=n_clips)
        clip_weights = np.bincount(sampled, minlength=n_clips)
        row_weights = clip_weights[clip_index].astype(float)
        total = row_weights.sum()
        value = 0.0
        for bin_id in range(5):
            mask = bin_index == bin_id
            weights = row_weights[mask]
            mass = weights.sum()
            if mass == 0:
                continue
            mean_confidence = np.average(confidence[mask], weights=weights)
            mean_correct = np.average(correct[mask], weights=weights)
            value += mass / total * abs(mean_confidence - mean_correct)
        samples[i] = value
        if (i + 1) % 100 == 0 or i + 1 == repeats:
            print(f"{label}: bootstrap sample {i + 1}/{repeats}", flush=True)
    return float(expected_calibration_error(group)), *np.percentile(samples, [2.5, 97.5])


def summarize(clip_scores: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, _ in RESPONSE_FILES:
        group = clip_scores[clip_scores["model_display"] == model]
        hce_items = scored[
            (scored["model_display"] == model)
            & (scored["task"] == "vhd_ii_hce")
            & scored["is_valid_response"]
        ]
        row = {"model_display": model}
        for metric in ["accuracy", "pdi"]:
            mean, low, high = bootstrap_ci(group[metric], f"{model} {metric}")
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        mean, low, high = bootstrap_hce_ci(hce_items, f"{model} hce")
        row["hce"] = mean
        row["hce_ci_low"] = low
        row["hce_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 72,
            "axes.labelsize": 72,
            "xtick.labelsize": 68,
            "ytick.labelsize": 68,
            "legend.fontsize": 64,
        }
    )
    metrics = [
        ("accuracy", "accuracy"),
        ("pdi", "pdi"),
        ("hce", "hce"),
    ]
    colors = ["#3E6F8E", "#D08C60", "#6A8F5E", "#8B6F9F"]
    saved_paths = []

    for metric, file_stem in metrics:
        single_path = FIGURE_DIR / f"{file_stem}.png"
        fig, axis = plt.subplots(figsize=(28.0, 15.5))
        values = summary[metric].to_numpy(dtype=float)
        low = summary[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = summary[f"{metric}_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([values - low, high - values])
        y = np.arange(len(summary))
        axis.barh(
            y,
            values,
            color=colors,
            height=0.62,
            edgecolor="#222222",
            linewidth=3.2,
        )
        axis.errorbar(
            values,
            y,
            xerr=xerr,
            fmt="none",
            ecolor="#111111",
            elinewidth=6.0,
            capsize=18,
            capthick=6.0,
        )
        for yy, value, upper in zip(y, values, high):
            label_x = max(upper + 0.055, value + 0.085)
            axis.text(label_x, yy, f"{value:.2f}", va="center", ha="left", fontsize=72)
        axis.set_yticks(y)
        axis.set_yticklabels(summary["model_display"].tolist())
        axis.invert_yaxis()
        label_limit = max(1.14, float(np.nanmax(high)) + 0.14)
        axis.set_xlim(0, label_limit)
        axis.set_xlabel("Score")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(4.8)
        axis.spines["bottom"].set_linewidth(4.8)
        axis.tick_params(axis="both", width=4.8, length=16)
        axis.grid(axis="x", color="#D0D0D0", linewidth=3.2)
        axis.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(single_path)
    return saved_paths


def main() -> None:
    global MANIFEST, BOOTSTRAP_REPEATS
    parser = argparse.ArgumentParser(description="Compute the main VHD metrics and figures.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="Path to the labelled evaluation_manifest.csv downloaded from Kaggle.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    MANIFEST = args.manifest.resolve()
    BOOTSTRAP_REPEATS = args.bootstrap_repeats
    if not MANIFEST.exists():
        parser.error(f"Manifest not found: {MANIFEST}")

    scored = score_items()
    clip_scores = per_clip_scores(scored)
    summary = summarize(clip_scores, scored)
    scored.to_csv(TABLE_DIR / "vhd_item_level_scored.csv", index=False)
    clip_scores.to_csv(TABLE_DIR / "vhd_clip_level_scores.csv", index=False)
    summary.to_csv(TABLE_DIR / "vhd_main_model_summary.csv", index=False)
    paths = plot(summary)
    print(summary.to_string(index=False))
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
