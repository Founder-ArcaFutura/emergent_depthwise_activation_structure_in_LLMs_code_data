import argparse
import glob
import os
import re
from io import StringIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INTERPOLATION_POINTS = 100
RAW_DATA_PATTERN = re.compile(
    r"--- RAW GRANULAR DATA \(Copy and Save as CSV\) ---\n(.*?)\n--- END OF RAW DATA ---",
    re.DOTALL,
)

MODEL_PARAMS = {
    "01-ai/Yi-1.5-9B": 9,
    "01-ai/Yi-1.5-9B-chat": 9,
    "01-ai/Yi-34B": 34,
    "01-ai/Yi-34B-Chat": 34,
    "01-ai/Yi-6B": 6,
    "01-ai/Yi-6B-Chat": 6,
    "01-ai/Yi-9B": 9,
    "01-ai/Yi-9B-200K": 9,
    "ByteDance/Ouro-2.6B-Thinking": 2.6,
    "CalderaAI/13B-Ouroboros": 13,
    "CalderaAI/13B-Theseus-MK1": 13,
    "EleutherAI/pythia-70m": 0.07,
    "EleutherAI/pythia-1.4b-deduped": 1.4,
    "EleutherAI/pythia-2.8b": 2.8,
    "EleutherAI/pythia-12b-deduped": 12,
    "HuggingFaceH4/zephyr-7b-beta": 7,
    "Qwen/QwQ-32B": 32,
    "Qwen/QwQ-32B-Preview": 32,
    "Qwen/Qwen2.5-14B-Instruct": 14,
    "Qwen/Qwen2.5-3B": 3,
    "Qwen/Qwen2.5-3B-Instruct": 3,
    "Qwen/Qwen3-1.7B": 1.7,
    "Qwen/Qwen3-1.7B-Base": 1.7,
    "Qwen/Qwen3-14B": 14,
    "Qwen/Qwen3-14B-Base": 14,
    "Qwen/Qwen3-4B": 4,
    "Qwen/Qwen3-8B-Base": 8,
    "TildeAI/TildeOpen-30b": 30,
    "WeiboAI/VibeThinker-1.5B": 1.5,
    "allenai/OLMo-2-1124-13B": 13,
    "cyberagent/calm3-22b-chat": 22,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 14,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 32,
    "deepseek-ai/deepseek-llm-7b-base": 7,
    "dphn/Dolphin-Mistral-24B-Venice-Edition": 24,
    "dphn/dolphin-2.9.1-yi-1.5-34b": 34,
    "lmsys/vicuna-7b-v1.5": 7,
    "meta-llama/Llama-3.1-8B": 8,
    "meta-llama/Llama-3.1-8B-Instruct": 8,
    "meta-llama/Llama-3.2-1B": 1,
    "meta-llama/Llama-3.2-3B": 3,
    "meta-llama/Llama-3.2-3B-Instruct": 3,
    "meta-llama/Meta-Llama-3-8B": 8,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8,
    "microsoft/Phi-4-reasoning-plus": 14,
    "microsoft/phi-2": 2.7,
    "microsoft/phi-4": 14,
    "mistralai/Mistral-7B-Instruct-v0.2": 7,
    "mistralai/Mistral-7B-Instruct-v0.3": 7,
    "mistralai/Mistral-7B-v0.1": 7,
    "mistralai/Mistral-7B-v0.3": 7,
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B": 1.5,
    "openai-community/gpt2": 0.12,
    "openai-community/gpt2-medium": 0.35,
    "openai-community/gpt2-xl": 1.5,
    "sshleifer/tiny-gpt2": 0.001,
    "stabilityai/StableBeluga-13B": 13,
    "stabilityai/StableBeluga-7B": 7,
    "stabilityai/stablelm-2-12b": 12,
    "stabilityai/stablelm-2-12b-chat": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified analysis for phase 1 and phase 3 result bundles.")
    parser.add_argument("--results-root", default="../results", help="Root directory containing phase result folders.")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory where plots/tables/reports will be written.")
    parser.add_argument("--min-params-b", type=float, default=3.0, help="Minimum parameter count in billions for filtered phase-3 analyses.")
    parser.add_argument("--phases", default="", help="Comma-separated phases to include, e.g. phase_1,phase_3.")
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA outputs.")
    parser.add_argument("--skip-report", action="store_true", help="Skip markdown report generation.")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip phase 1 invariance analyses.")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip phase 3 analyses.")
    return parser.parse_args()


def parse_metadata_from_id(model_id):
    model_id_lower = str(model_id).lower()
    if any(token in model_id_lower for token in ["instruct", "chat", "rl", "it", "sft"]):
        alignment = "instruct"
    else:
        alignment = "base"

    if "llama" in model_id_lower:
        family = "LLaMA"
    elif "mistral" in model_id_lower or "mixtral" in model_id_lower:
        family = "Mistral"
    elif "qwen" in model_id_lower:
        family = "Qwen"
    elif "yi-" in model_id_lower:
        family = "Yi"
    elif "deepseek" in model_id_lower:
        family = "DeepSeek"
    elif "olmo" in model_id_lower:
        family = "OLMo"
    elif "pythia" in model_id_lower:
        family = "Pythia"
    elif "phi" in model_id_lower:
        family = "Phi"
    elif "stablebeluga" in model_id_lower or "stablelm" in model_id_lower:
        family = "StabilityAI"
    else:
        family = "Other"
    return family, alignment


def infer_phase_metadata(path):
    parts = path.parts
    phase = None
    run_group = None
    hardware_group = None
    prompt_set = None

    for index, part in enumerate(parts):
        if part.startswith("phase_"):
            phase = part
            if phase == "phase_1" and index + 1 < len(parts):
                run_group = parts[index + 1]
            break

    if run_group:
        run_group_lower = run_group.lower()
        if "a10g" in run_group_lower:
            hardware_group = "A10G"
        elif "1xl40s" in run_group_lower:
            hardware_group = "1xL40S"
        prompt_match = re.search(r"(prompts_\d+)", run_group_lower)
        if prompt_match:
            prompt_set = prompt_match.group(1)

    return phase, run_group, hardware_group, prompt_set


def extract_training_step(path, revision_value):
    if pd.notna(revision_value):
        revision_match = re.search(r"step(\d+)", str(revision_value))
        if revision_match:
            return int(revision_match.group(1))
        if str(revision_value).lower() == "main":
            return 0

    filename_match = re.search(r"step(\d+)", path.name)
    if filename_match:
        return int(filename_match.group(1))
    return 0


def ensure_directories(output_dir):
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    pca_dir = os.path.join(output_dir, "pca")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)
    return plots_dir, tables_dir, pca_dir


def load_single_markdown_result(path):
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  - ERROR: Could not read {path}: {exc}")
        return None

    match = RAW_DATA_PATTERN.search(content)
    if not match:
        return None

    try:
        df = pd.read_csv(StringIO(match.group(1)))
    except Exception as exc:
        print(f"  - ERROR: Failed parsing CSV block in {path}: {exc}")
        return None

    required_columns = {"model_id", "prompt", "prompt_type", "layer_index", "activation_norm", "experiment_timestamp"}
    if not required_columns.issubset(df.columns):
        return None

    df["activation_norm"] = pd.to_numeric(df["activation_norm"], errors="coerce")
    df["layer_index"] = pd.to_numeric(df["layer_index"], errors="coerce")
    df = df.dropna(subset=["activation_norm", "layer_index"])
    if df.empty:
        return None

    phase, run_group, hardware_group, prompt_set = infer_phase_metadata(path)
    model_id = df["model_id"].iloc[0]
    family, alignment = parse_metadata_from_id(model_id)
    revision_value = df["revision"].iloc[0] if "revision" in df.columns else np.nan
    training_step = extract_training_step(path, revision_value)

    df["phase"] = phase
    df["run_group"] = run_group
    df["hardware_group"] = hardware_group
    df["prompt_set"] = prompt_set
    df["training_step"] = training_step
    df["model_family"] = family
    df["alignment_status"] = alignment
    df["params_b"] = MODEL_PARAMS.get(model_id, np.nan)
    df["source_file"] = str(path)
    df["curve_id"] = (
        df["phase"].fillna("unknown").astype(str)
        + "|"
        + df["model_id"].astype(str)
        + "|"
        + df["prompt"].astype(str)
        + "|"
        + df["experiment_timestamp"].astype(str)
        + "|"
        + df["source_file"].astype(str)
    )
    return df


def normalize_layers(group):
    max_layer = group["layer_index"].max()
    group = group.copy()
    if max_layer > 0:
        group["layer_index_norm"] = group["layer_index"] / max_layer
    else:
        group["layer_index_norm"] = 0.0
    return group


def load_all_results(results_root, requested_phases):
    all_files = [Path(fp) for fp in glob.glob(os.path.join(results_root, "**", "*.md"), recursive=True)]
    if not all_files:
        raise FileNotFoundError(f"No markdown result files found under {results_root}")

    frames = []
    for path in tqdm(sorted(all_files), desc="Loading result files"):
        if path.name.lower() == "blank.md":
            continue
        df = load_single_markdown_result(path)
        if df is None:
            continue
        if requested_phases and df["phase"].iloc[0] not in requested_phases:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No usable result files were loaded from the selected phases.")

    master_df = pd.concat(frames, ignore_index=True)
    max_layers = master_df.groupby("curve_id")["layer_index"].transform("max")
    master_df["layer_index_norm"] = np.where(max_layers > 0, master_df["layer_index"] / max_layers, 0.0)
    return master_df


def zscore(values):
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def interpolate_curve(group, x_interp, value_column="activation_norm", normalize=False):
    group = group.drop_duplicates(subset=["layer_index_norm"]).sort_values("layer_index_norm")
    if len(group) < 2:
        return None
    values = group[value_column].to_numpy(dtype=float)
    if normalize:
        values = zscore(values)
    interpolator = interp1d(group["layer_index_norm"].to_numpy(dtype=float), values, kind="linear", fill_value="extrapolate")
    return interpolator(x_interp)


def build_mean_curves(df, x_interp):
    records = []
    for keys, group in df.groupby(["model_id", "phase", "prompt_type", "curve_id"]):
        curve = interpolate_curve(group, x_interp, normalize=True)
        raw_curve = interpolate_curve(group, x_interp, normalize=False)
        if curve is None or raw_curve is None:
            continue
        model_id, phase, prompt_type, curve_id = keys
        records.append(
            {
                "model_id": model_id,
                "phase": phase,
                "prompt_type": prompt_type,
                "curve_id": curve_id,
                "z_curve": curve,
                "raw_curve": raw_curve,
                "model_family": group["model_family"].iloc[0],
                "alignment_status": group["alignment_status"].iloc[0],
                "params_b": group["params_b"].iloc[0],
                "training_step": group["training_step"].iloc[0],
                "hardware_group": group["hardware_group"].iloc[0],
                "prompt_set": group["prompt_set"].iloc[0],
                "run_group": group["run_group"].iloc[0],
            }
        )
    curve_df = pd.DataFrame(records)
    if curve_df.empty:
        return curve_df, pd.DataFrame()

    mean_records = []
    group_cols = [
        "model_id",
        "phase",
        "prompt_type",
        "model_family",
        "alignment_status",
        "params_b",
        "training_step",
        "hardware_group",
        "prompt_set",
        "run_group",
    ]
    for keys, group in curve_df.groupby(group_cols, dropna=False):
        mean_records.append(
            {
                **dict(zip(group_cols, keys)),
                "mean_z_curve": np.mean(np.vstack(group["z_curve"]), axis=0),
                "mean_raw_curve": np.mean(np.vstack(group["raw_curve"]), axis=0),
            }
        )
    return curve_df, pd.DataFrame(mean_records)


def compute_prompt_metrics(df):
    cleaned = df.dropna(subset=["activation_norm"]).copy()
    cleaned = cleaned[cleaned["activation_norm"] > 0]
    if cleaned.empty:
        return pd.DataFrame()

    cleaned["normalized_activation_norm"] = cleaned.groupby("curve_id")["activation_norm"].transform(
        lambda values: values / values.sum() if values.sum() > 0 else values
    )

    records = []
    for keys, group in cleaned.groupby(["model_id", "phase", "curve_id", "prompt_type"], dropna=False):
        weights = group["normalized_activation_norm"].to_numpy(dtype=float)
        positions = group["layer_index_norm"].to_numpy(dtype=float)
        denom = weights.sum()
        if denom <= 0:
            wli = 0.0
            wsd = 0.0
        else:
            wli = float(np.sum(positions * weights) / denom)
            variance = float(np.sum(weights * (positions - wli) ** 2) / denom)
            wsd = float(np.sqrt(max(variance, 0.0)))
        model_id, phase, curve_id, prompt_type = keys
        records.append(
            {
                "model_id": model_id,
                "phase": phase,
                "curve_id": curve_id,
                "prompt_type": prompt_type,
                "wli": wli,
                "wsd": wsd,
                "model_family": group["model_family"].iloc[0],
                "alignment_status": group["alignment_status"].iloc[0],
                "params_b": group["params_b"].iloc[0],
                "training_step": group["training_step"].iloc[0],
                "hardware_group": group["hardware_group"].iloc[0],
                "prompt_set": group["prompt_set"].iloc[0],
                "run_group": group["run_group"].iloc[0],
            }
        )
    return pd.DataFrame(records)


def compute_model_delta_phi(prompt_metrics_df, grouping_columns):
    if prompt_metrics_df.empty:
        return pd.DataFrame()

    grouped_metrics = []
    for keys, group in prompt_metrics_df.groupby(grouping_columns, dropna=False):
        dense = group[group["prompt_type"] == "dense"]
        flat = group[group["prompt_type"] == "flat"]
        if dense.empty or flat.empty:
            continue
        dense_wli = float(dense["wli"].mean())
        flat_wli = float(flat["wli"].mean())
        dense_wsd = float(dense["wsd"].mean())
        flat_wsd = float(flat["wsd"].mean())
        row = dict(zip(grouping_columns, keys))
        row.update(
            {
                "wli_dense": dense_wli,
                "wli_flat": flat_wli,
                "wsd_dense": dense_wsd,
                "wsd_flat": flat_wsd,
                "delta": dense_wli - flat_wli,
                "phi": flat_wsd - dense_wsd,
            }
        )
        grouped_metrics.append(row)
    return pd.DataFrame(grouped_metrics)


def safe_curve_correlation(curve_a, curve_b):
    if curve_a is None or curve_b is None:
        return np.nan
    if np.std(curve_a) < 1e-9 and np.std(curve_b) < 1e-9:
        return 1.0
    if np.std(curve_a) < 1e-9 or np.std(curve_b) < 1e-9:
        return 0.0
    return float(np.corrcoef(curve_a, curve_b)[0, 1])


def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def run_phase1_hardware_invariance(master_df, mean_curve_df, prompt_metrics_df, plots_dir, tables_dir, x_interp):
    phase1_df = master_df[master_df["phase"] == "phase_1"].copy()
    if phase1_df.empty:
        return None

    phase1_metrics = compute_model_delta_phi(
        prompt_metrics_df[prompt_metrics_df["phase"] == "phase_1"],
        ["model_id", "hardware_group", "prompt_set", "model_family", "alignment_status", "params_b"],
    )
    if phase1_metrics.empty:
        return None

    phase1_means = mean_curve_df[mean_curve_df["phase"] == "phase_1"].copy()
    rows = []
    for (model_id, prompt_set), subset in phase1_means.groupby(["model_id", "prompt_set"], dropna=False):
        hardware_values = set(subset["hardware_group"].dropna())
        if {"A10G", "1xL40S"} - hardware_values:
            continue

        for prompt_type in ["dense", "flat"]:
            typed_subset = subset[subset["prompt_type"] == prompt_type]
            a10g_row = typed_subset[typed_subset["hardware_group"] == "A10G"]
            l40_row = typed_subset[typed_subset["hardware_group"] == "1xL40S"]
            if a10g_row.empty or l40_row.empty:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "prompt_set": prompt_set,
                    "prompt_type": prompt_type,
                    "curve_similarity": safe_curve_correlation(
                        a10g_row["mean_z_curve"].iloc[0],
                        l40_row["mean_z_curve"].iloc[0],
                    ),
                    "model_family": a10g_row["model_family"].iloc[0],
                    "alignment_status": a10g_row["alignment_status"].iloc[0],
                }
            )

    similarity_df = pd.DataFrame(rows)
    merged_metrics = phase1_metrics.pivot_table(
        index=["model_id", "prompt_set", "model_family", "alignment_status", "params_b"],
        columns="hardware_group",
        values=["delta", "wli_dense", "wli_flat", "phi"],
        aggfunc="mean",
    )
    if merged_metrics.empty:
        return None

    merged_metrics.columns = [f"{metric}_{hardware}" for metric, hardware in merged_metrics.columns]
    merged_metrics = merged_metrics.reset_index()
    merged_metrics["delta_difference"] = merged_metrics.get("delta_A10G", np.nan) - merged_metrics.get("delta_1xL40S", np.nan)
    merged_metrics["wli_dense_difference"] = merged_metrics.get("wli_dense_A10G", np.nan) - merged_metrics.get("wli_dense_1xL40S", np.nan)
    similarity_summary = similarity_df.groupby(["model_id", "prompt_set"], as_index=False)["curve_similarity"].mean()
    merged_metrics = merged_metrics.merge(similarity_summary, on=["model_id", "prompt_set"], how="left")

    table_path = os.path.join(tables_dir, "phase1_hardware_invariance.csv")
    save_dataframe(merged_metrics, table_path)

    if not merged_metrics.empty:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].scatter(merged_metrics["delta_A10G"], merged_metrics["delta_1xL40S"], alpha=0.8)
        diagonal = [
            np.nanmin([merged_metrics["delta_A10G"].min(), merged_metrics["delta_1xL40S"].min()]),
            np.nanmax([merged_metrics["delta_A10G"].max(), merged_metrics["delta_1xL40S"].max()]),
        ]
        axes[0].plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1)
        axes[0].set_title("Phase 1 delta agreement by hardware")
        axes[0].set_xlabel("Delta on A10G")
        axes[0].set_ylabel("Delta on 1xL40S")

        similarity_plot = similarity_df.pivot_table(index="model_id", columns="prompt_set", values="curve_similarity", aggfunc="mean")
        sns.heatmap(similarity_plot, cmap="viridis", vmin=0, vmax=1, ax=axes[1])
        axes[1].set_title("Phase 1 hardware curve similarity")
        axes[1].set_xlabel("Prompt set")
        axes[1].set_ylabel("Model")
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "phase1_hardware_invariance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

        matched_models = merged_metrics["model_id"].dropna().unique()[:6]
        if len(matched_models) > 0:
            fig, axes = plt.subplots(len(matched_models), 1, figsize=(12, 3 * len(matched_models)), sharex=True)
            if len(matched_models) == 1:
                axes = [axes]
            for ax, model_id in zip(axes, matched_models):
                subset = phase1_means[
                    (phase1_means["model_id"] == model_id)
                    & (phase1_means["prompt_set"] == "prompts_1")
                    & (phase1_means["prompt_type"] == "dense")
                ]
                for hardware_group, line_style in [("A10G", "-"), ("1xL40S", "--")]:
                    row = subset[subset["hardware_group"] == hardware_group]
                    if row.empty:
                        continue
                    ax.plot(x_interp, row["mean_z_curve"].iloc[0], line_style, linewidth=2, label=hardware_group)
                ax.set_title(model_id)
                ax.set_ylabel("Z-scored activation")
                ax.legend()
            axes[-1].set_xlabel("Normalized layer index")
            plt.tight_layout()
            overlay_path = os.path.join(plots_dir, "phase1_hardware_overlay.png")
            plt.savefig(overlay_path, dpi=300)
            plt.close()
            print(f"Saved plot: {overlay_path}")

    summary = {
        "table_path": table_path,
        "num_models": int(merged_metrics["model_id"].nunique()),
        "mean_curve_similarity": float(similarity_df["curve_similarity"].mean()) if not similarity_df.empty else np.nan,
        "median_abs_delta_difference": float(merged_metrics["delta_difference"].abs().median()) if not merged_metrics.empty else np.nan,
        "worst_abs_delta_difference": float(merged_metrics["delta_difference"].abs().max()) if not merged_metrics.empty else np.nan,
        "dataframe": merged_metrics,
    }
    return summary


def run_phase1_prompt_invariance(master_df, mean_curve_df, prompt_metrics_df, plots_dir, tables_dir, x_interp):
    phase1_df = master_df[master_df["phase"] == "phase_1"].copy()
    if phase1_df.empty:
        return None

    phase1_metrics = compute_model_delta_phi(
        prompt_metrics_df[prompt_metrics_df["phase"] == "phase_1"],
        ["model_id", "prompt_set", "hardware_group", "model_family", "alignment_status", "params_b"],
    )
    if phase1_metrics.empty:
        return None

    phase1_means = mean_curve_df[mean_curve_df["phase"] == "phase_1"].copy()
    records = []
    for (model_id, hardware_group), subset in phase1_means.groupby(["model_id", "hardware_group"], dropna=False):
        prompt_values = set(subset["prompt_set"].dropna())
        if {"prompts_1", "prompts_2"} - prompt_values:
            continue
        row = {
            "model_id": model_id,
            "hardware_group": hardware_group,
            "model_family": subset["model_family"].iloc[0],
            "alignment_status": subset["alignment_status"].iloc[0],
        }
        for prompt_type in ["dense", "flat"]:
            typed_subset = subset[subset["prompt_type"] == prompt_type]
            p1 = typed_subset[typed_subset["prompt_set"] == "prompts_1"]
            p2 = typed_subset[typed_subset["prompt_set"] == "prompts_2"]
            if p1.empty or p2.empty:
                continue
            row[f"{prompt_type}_similarity"] = safe_curve_correlation(p1["mean_z_curve"].iloc[0], p2["mean_z_curve"].iloc[0])

        p1_dense = subset[(subset["prompt_set"] == "prompts_1") & (subset["prompt_type"] == "dense")]
        p1_flat = subset[(subset["prompt_set"] == "prompts_1") & (subset["prompt_type"] == "flat")]
        p2_dense = subset[(subset["prompt_set"] == "prompts_2") & (subset["prompt_type"] == "dense")]
        p2_flat = subset[(subset["prompt_set"] == "prompts_2") & (subset["prompt_type"] == "flat")]
        if not p1_dense.empty and not p1_flat.empty and not p2_dense.empty and not p2_flat.empty:
            delta_curve_1 = p1_dense["mean_z_curve"].iloc[0] - p1_flat["mean_z_curve"].iloc[0]
            delta_curve_2 = p2_dense["mean_z_curve"].iloc[0] - p2_flat["mean_z_curve"].iloc[0]
            row["delta_curve_similarity"] = safe_curve_correlation(delta_curve_1, delta_curve_2)
        records.append(row)

    similarity_df = pd.DataFrame(records)
    merged_metrics = phase1_metrics.pivot_table(
        index=["model_id", "hardware_group", "model_family", "alignment_status", "params_b"],
        columns="prompt_set",
        values=["delta", "wli_dense", "wli_flat", "phi"],
        aggfunc="mean",
    )
    if merged_metrics.empty:
        return None

    merged_metrics.columns = [f"{metric}_{prompt_set}" for metric, prompt_set in merged_metrics.columns]
    merged_metrics = merged_metrics.reset_index()
    merged_metrics["delta_difference"] = merged_metrics.get("delta_prompts_1", np.nan) - merged_metrics.get("delta_prompts_2", np.nan)
    merged_metrics = merged_metrics.merge(similarity_df, on=["model_id", "hardware_group", "model_family", "alignment_status"], how="left")
    table_path = os.path.join(tables_dir, "phase1_prompt_invariance.csv")
    save_dataframe(merged_metrics, table_path)

    if not merged_metrics.empty:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].scatter(merged_metrics["delta_prompts_1"], merged_metrics["delta_prompts_2"], alpha=0.8)
        diagonal = [
            np.nanmin([merged_metrics["delta_prompts_1"].min(), merged_metrics["delta_prompts_2"].min()]),
            np.nanmax([merged_metrics["delta_prompts_1"].max(), merged_metrics["delta_prompts_2"].max()]),
        ]
        axes[0].plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1)
        axes[0].set_title("Phase 1 delta agreement by prompt set")
        axes[0].set_xlabel("Delta on prompts_1")
        axes[0].set_ylabel("Delta on prompts_2")

        if not similarity_df.empty:
            long_df = similarity_df.melt(
                id_vars=["model_id", "hardware_group"],
                value_vars=[col for col in ["dense_similarity", "flat_similarity", "delta_curve_similarity"] if col in similarity_df.columns],
                var_name="metric",
                value_name="similarity",
            )
            sns.boxplot(data=long_df, x="metric", y="similarity", ax=axes[1])
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Phase 1 prompt-set curve similarity")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("Correlation")
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "phase1_prompt_invariance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

        matched_models = merged_metrics["model_id"].dropna().unique()[:6]
        if len(matched_models) > 0:
            fig, axes = plt.subplots(len(matched_models), 1, figsize=(12, 3 * len(matched_models)), sharex=True)
            if len(matched_models) == 1:
                axes = [axes]
            for ax, model_id in zip(axes, matched_models):
                subset = phase1_means[
                    (phase1_means["model_id"] == model_id)
                    & (phase1_means["hardware_group"] == "A10G")
                    & (phase1_means["prompt_type"] == "dense")
                ]
                for prompt_set, line_style in [("prompts_1", "-"), ("prompts_2", "--")]:
                    row = subset[subset["prompt_set"] == prompt_set]
                    if row.empty:
                        continue
                    ax.plot(x_interp, row["mean_z_curve"].iloc[0], line_style, linewidth=2, label=prompt_set)
                ax.set_title(model_id)
                ax.set_ylabel("Z-scored activation")
                ax.legend()
            axes[-1].set_xlabel("Normalized layer index")
            plt.tight_layout()
            overlay_path = os.path.join(plots_dir, "phase1_prompt_overlay.png")
            plt.savefig(overlay_path, dpi=300)
            plt.close()
            print(f"Saved plot: {overlay_path}")

    rank_source = merged_metrics[["model_id", "delta_prompts_1", "delta_prompts_2"]].dropna()
    rank_corr = np.nan
    if len(rank_source) >= 2:
        rank_corr = float(rank_source["delta_prompts_1"].rank().corr(rank_source["delta_prompts_2"].rank(), method="spearman"))

    summary = {
        "table_path": table_path,
        "num_models": int(merged_metrics["model_id"].nunique()),
        "mean_delta_curve_similarity": float(similarity_df["delta_curve_similarity"].mean()) if "delta_curve_similarity" in similarity_df else np.nan,
        "prompt_rank_spearman": rank_corr,
        "dataframe": merged_metrics,
    }
    return summary


def run_phase3_normative_overlay(master_df, mean_curve_df, plots_dir, tables_dir, x_interp, min_params_b):
    eligible = mean_curve_df[
        mean_curve_df["phase"].isin(["phase_3", "phase_4"]) & (mean_curve_df["training_step"] == 0)
    ].copy()
    eligible = eligible[eligible["params_b"].fillna(-1) >= min_params_b]
    if eligible.empty:
        return None

    all_curves = []
    curve_models = []
    for _, row in eligible.iterrows():
        all_curves.append(row["mean_z_curve"])
        curve_models.append(row["model_id"])

    if not all_curves:
        return None

    curves = np.vstack(all_curves)
    curve_models = np.array(curve_models)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    deviation_records = []
    for model_id in sorted(set(curve_models)):
        model_curves = curves[curve_models == model_id]
        deviation_records.append(
            {
                "model_id": model_id,
                "model_family": eligible[eligible["model_id"] == model_id]["model_family"].iloc[0],
                "alignment_status": eligible[eligible["model_id"] == model_id]["alignment_status"].iloc[0],
                "params_b": eligible[eligible["model_id"] == model_id]["params_b"].iloc[0],
                "norm_band_deviation": float(np.mean(np.abs(model_curves - mean_curve) / (std_curve + 1e-9))),
            }
        )

    deviation_df = pd.DataFrame(deviation_records).sort_values("norm_band_deviation", ascending=False)
    table_path = os.path.join(tables_dir, "model_deviation_scores.csv")
    save_dataframe(deviation_df, table_path)

    top_models = deviation_df["model_id"].head(8).tolist()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))
    for curve, model_id in zip(curves, curve_models):
        ax.plot(x_interp, curve, color="grey", alpha=0.06, linewidth=0.7)
    for color, model_id in zip(sns.color_palette("bright", len(top_models)), top_models):
        model_curves = curves[curve_models == model_id]
        ax.plot(x_interp, model_curves.mean(axis=0), color=color, linewidth=2.5, label=model_id.split("/")[-1])
    ax.fill_between(x_interp, mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.12, label="Std. dev.")
    ax.plot(x_interp, mean_curve, color="black", alpha=0.25, linewidth=6)
    ax.plot(x_interp, mean_curve, color="black", linestyle="--", linewidth=2.5, label="Mean activation curve")
    ax.set_title("Normative tri-phasic activation pattern")
    ax.set_xlabel("Normalized layer index")
    ax.set_ylabel("Z-scored activation")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    normative_path = os.path.join(plots_dir, "normative_tri_phasic_activation.png")
    plt.savefig(normative_path, dpi=300)
    plt.close()
    print(f"Saved plot: {normative_path}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for prompt_type, color, ax in [("dense", "#1f77b4", axes[0]), ("flat", "#d62728", axes[1])]:
        subset = eligible[eligible["prompt_type"] == prompt_type]
        for curve in subset["mean_z_curve"]:
            ax.plot(x_interp, curve, color=color, alpha=0.08, linewidth=0.8)
        if not subset.empty:
            mean_prompt_curve = np.mean(np.vstack(subset["mean_z_curve"]), axis=0)
            ax.plot(x_interp, mean_prompt_curve, color="black", linestyle="--", linewidth=2.5)
        ax.set_title(f"Overlay: {prompt_type}")
        ax.set_xlabel("Normalized layer index")
        ax.set_ylabel("Z-scored activation")
    plt.tight_layout()
    overlay_path = os.path.join(plots_dir, "phase3_overlay_curves.png")
    plt.savefig(overlay_path, dpi=300)
    plt.close()
    print(f"Saved plot: {overlay_path}")

    pca_candidates = []
    phase_rank = {"phase_4": 4, "phase_3": 3, "phase_2_statements_only": 2, "phase_1": 1}
    for model_id, group in mean_curve_df[
        mean_curve_df["phase"].isin(["phase_3", "phase_4"]) & (mean_curve_df["training_step"] == 0)
    ].groupby("model_id"):
        dense_rows = group[group["prompt_type"] == "dense"].copy()
        flat_rows = group[group["prompt_type"] == "flat"].copy()
        if dense_rows.empty or flat_rows.empty:
            continue
        dense_rows["phase_rank"] = dense_rows["phase"].map(phase_rank).fillna(0)
        flat_rows["phase_rank"] = flat_rows["phase"].map(phase_rank).fillna(0)
        best_dense = dense_rows.sort_values(["phase_rank", "params_b"], ascending=[False, False]).iloc[0]
        best_flat = flat_rows.sort_values(["phase_rank", "params_b"], ascending=[False, False]).iloc[0]
        pca_candidates.append(
            {
                "model_id": model_id,
                "model_family": best_dense["model_family"],
                "alignment_status": best_dense["alignment_status"],
                "params_b": best_dense["params_b"],
                "dense_curve": best_dense["mean_z_curve"],
                "flat_curve": best_flat["mean_z_curve"],
            }
        )

    return {
        "table_path": table_path,
        "normative_plot_path": normative_path,
        "overlay_plot_path": overlay_path,
        "deviation_df": deviation_df,
        "pca_candidate_df": pd.DataFrame(pca_candidates),
    }


def run_phase3_pca(normative_result, plots_dir, pca_dir, x_interp):
    if normative_result is None:
        return None

    candidate_df = normative_result["pca_candidate_df"].copy()
    if candidate_df.empty or len(candidate_df) < 2:
        return None

    dense_matrix = np.vstack(candidate_df["dense_curve"].to_numpy())
    flat_matrix = np.vstack(candidate_df["flat_curve"].to_numpy())
    delta_matrix = dense_matrix - flat_matrix

    def run_pc1(matrix):
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(centered)[:, 0]
        component = pca.components_[0]
        variance = float(pca.explained_variance_ratio_[0])
        if component[0] < 0:
            component = -component
            scores = -scores
        return component, variance, scores

    dense_pc1, dense_var, dense_scores = run_pc1(dense_matrix)
    flat_pc1, flat_var, flat_scores = run_pc1(flat_matrix)
    delta_pc1, delta_var, delta_scores = run_pc1(delta_matrix)

    variance_df = pd.DataFrame(
        {
            "curve_type": ["dense", "flat", "delta"],
            "pc1_explained_variance_ratio": [dense_var, flat_var, delta_var],
        }
    )
    save_dataframe(variance_df, os.path.join(pca_dir, "variance_explained.csv"))

    components_df = pd.DataFrame(
        {
            "x": x_interp,
            "dense_pc1": dense_pc1,
            "flat_pc1": flat_pc1,
            "delta_pc1": delta_pc1,
        }
    )
    save_dataframe(components_df, os.path.join(pca_dir, "canonical_curves_60.csv"))

    score_df = candidate_df[["model_id", "model_family", "alignment_status", "params_b"]].copy()
    score_df["dense_pc1_score"] = dense_scores
    score_df["flat_pc1_score"] = flat_scores
    score_df["delta_pc1_score"] = delta_scores
    save_dataframe(score_df, os.path.join(pca_dir, "model_level_pc1_scores.csv"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_interp, dense_pc1, linewidth=3, label=f"Dense PC1 ({dense_var:.1%})", color="blue")
    ax.plot(x_interp, flat_pc1, linewidth=3, label=f"Flat PC1 ({flat_var:.1%})", color="green")
    ax.plot(x_interp, delta_pc1, linewidth=3, label=f"Delta PC1 ({delta_var:.1%})", color="red")
    ax.set_title(f"Triphasic activation PCA overlay (N={len(candidate_df)} models)")
    ax.set_xlabel("Normalized layer index")
    ax.set_ylabel("Component loading")
    ax.legend()
    plt.tight_layout()
    components_path = os.path.join(plots_dir, "canonical_pc1_curves_60.png")
    plt.savefig(components_path, dpi=300)
    plt.close()
    print(f"Saved plot: {components_path}")

    return {
        "variance_df": variance_df,
        "score_df": score_df,
        "components_path": components_path,
    }


def run_phase3_training_emergence(master_df, prompt_metrics_df, plots_dir, tables_dir):
    phase3_df = master_df[(master_df["phase"] == "phase_3") & (master_df["training_step"] > 0)].copy()
    if phase3_df.empty:
        return None

    prompt_metrics = prompt_metrics_df[(prompt_metrics_df["phase"] == "phase_3") & (prompt_metrics_df["training_step"] > 0)]
    trajectory_df = compute_model_delta_phi(
        prompt_metrics,
        ["model_id", "model_family", "training_step", "params_b"],
    )
    trajectory_df = trajectory_df[trajectory_df["model_family"].isin(["OLMo", "Pythia"])]
    if trajectory_df.empty:
        return None

    table_path = os.path.join(tables_dir, "training_emergence_metrics.csv")
    save_dataframe(trajectory_df, table_path)

    for family in sorted(trajectory_df["model_family"].unique()):
        family_metrics = trajectory_df[trajectory_df["model_family"] == family].sort_values("training_step")
        family_df = phase3_df[phase3_df["model_family"] == family].copy()
        if family_metrics.empty or family_df.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        sns.lineplot(data=family_metrics, x="training_step", y="wli_dense", marker="o", ax=axes[0], label="Dense WLI")
        sns.lineplot(data=family_metrics, x="training_step", y="wli_flat", marker="o", ax=axes[0], label="Flat WLI")
        axes[0].set_title(f"{family} WLI over training")
        axes[0].set_ylabel("WLI")

        sns.lineplot(data=family_metrics, x="training_step", y="delta", marker="o", ax=axes[1], color="purple")
        axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
        axes[1].set_title(f"{family} delta over training")
        axes[1].set_ylabel("Delta")
        axes[1].set_xlabel("Training step")
        plt.tight_layout()
        metrics_path = os.path.join(plots_dir, f"developmental_metrics_{family}.png")
        plt.savefig(metrics_path, dpi=300)
        plt.close()
        print(f"Saved plot: {metrics_path}")

        dense_curves = []
        curve_labels = []
        for step, step_group in family_df[family_df["prompt_type"] == "dense"].groupby("training_step"):
            mean_curve = step_group.groupby("layer_index_norm")["activation_norm"].mean().sort_index()
            if len(mean_curve) < 2:
                continue
            dense_curves.append(zscore(mean_curve.to_numpy()))
            curve_labels.append(step)
        if dense_curves:
            x_values = np.linspace(0.0, 1.0, dense_curves[0].shape[0])
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(dense_curves)))
            for color, step, curve in zip(colors, curve_labels, dense_curves):
                plt.plot(x_values, curve, color=color, linewidth=2, alpha=0.85, label=f"Step {step}")
            plt.title(f"{family} dense activation evolution")
            plt.xlabel("Normalized layer index")
            plt.ylabel("Z-scored activation")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            curves_path = os.path.join(plots_dir, f"developmental_curves_{family}.png")
            plt.savefig(curves_path, dpi=300)
            plt.close()
            print(f"Saved plot: {curves_path}")

    return {"table_path": table_path, "trajectory_df": trajectory_df}


def generate_markdown_report(output_dir, master_df, phase1_hardware, phase1_prompt, normative_result, pca_result, training_result):
    report_path = os.path.join(output_dir, "analysis_report.md")
    coverage = master_df.groupby("phase")["model_id"].nunique().reset_index(name="unique_models").sort_values("phase")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Unified Analysis Report\n\n")
        handle.write("This report summarizes the analyses reconstructed from the bundled result logs. It is descriptive and limited to what can be computed directly from the current repository artifacts.\n\n")

        handle.write("## Data Coverage\n\n")
        handle.write(coverage.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Phase 1 Hardware Invariance\n\n")
        if phase1_hardware is None:
            handle.write("No matched hardware comparisons were available.\n\n")
        else:
            handle.write(f"Matched models: {phase1_hardware['num_models']}\n\n")
            handle.write(f"- Mean curve similarity: {phase1_hardware['mean_curve_similarity']:.4f}\n")
            handle.write(f"- Median absolute delta difference: {phase1_hardware['median_abs_delta_difference']:.4f}\n")
            handle.write(f"- Worst absolute delta difference: {phase1_hardware['worst_abs_delta_difference']:.4f}\n\n")
            handle.write("![Hardware invariance](plots/phase1_hardware_invariance.png)\n\n")
            handle.write("![Hardware overlay](plots/phase1_hardware_overlay.png)\n\n")

        handle.write("## Phase 1 Prompt-Pattern Invariance\n\n")
        if phase1_prompt is None:
            handle.write("No matched prompt-set comparisons were available.\n\n")
        else:
            handle.write(f"Matched models: {phase1_prompt['num_models']}\n\n")
            if pd.notna(phase1_prompt["mean_delta_curve_similarity"]):
                handle.write(f"- Mean delta-curve similarity: {phase1_prompt['mean_delta_curve_similarity']:.4f}\n")
            if pd.notna(phase1_prompt["prompt_rank_spearman"]):
                handle.write(f"- Delta rank Spearman correlation: {phase1_prompt['prompt_rank_spearman']:.4f}\n")
            handle.write("\n![Prompt invariance](plots/phase1_prompt_invariance.png)\n\n")
            handle.write("![Prompt overlay](plots/phase1_prompt_overlay.png)\n\n")

        handle.write("## Phase 3 Normative And Overlay Analyses\n\n")
        if normative_result is None:
            handle.write("No phase 3 main-model overlay analysis was generated.\n\n")
        else:
            handle.write("![Normative curve](plots/normative_tri_phasic_activation.png)\n\n")
            handle.write("![Overlay curves](plots/phase3_overlay_curves.png)\n\n")
            top_outliers = normative_result["deviation_df"].head(10)
            if not top_outliers.empty:
                handle.write("### Highest deviation models\n\n")
                handle.write(top_outliers.to_markdown(index=False, floatfmt=".4f"))
                handle.write("\n\n")

        handle.write("## Phase 3 PCA\n\n")
        if pca_result is None:
            handle.write("PCA was skipped or there were not enough model-level curves.\n\n")
        else:
            handle.write("![Triphasic PCA overlay](plots/canonical_pc1_curves_60.png)\n\n")
            handle.write(pca_result["variance_df"].to_markdown(index=False, floatfmt=".4f"))
            handle.write("\n\n")

        handle.write("## Phase 3 Training Emergence\n\n")
        if training_result is None:
            handle.write("No checkpoint trajectories were available.\n\n")
        else:
            for family in sorted(training_result["trajectory_df"]["model_family"].unique()):
                metrics_filename = f"plots/developmental_metrics_{family}.png"
                curves_filename = f"plots/developmental_curves_{family}.png"
                handle.write(f"### {family}\n\n")
                if os.path.exists(os.path.join(output_dir, metrics_filename)):
                    handle.write(f"![{family} metrics]({metrics_filename})\n\n")
                if os.path.exists(os.path.join(output_dir, curves_filename)):
                    handle.write(f"![{family} curves]({curves_filename})\n\n")

    print(f"Saved report: {report_path}")
    return report_path


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_root = os.path.normpath(os.path.join(script_dir, args.results_root))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    plots_dir, tables_dir, pca_dir = ensure_directories(output_dir)
    requested_phases = {phase.strip() for phase in args.phases.split(",") if phase.strip()}

    sns.set_theme(style="whitegrid")
    print(f"Loading results from: {results_root}")
    master_df = load_all_results(results_root, requested_phases)
    print(
        "Loaded "
        f"{master_df['model_id'].nunique()} models, "
        f"{master_df['curve_id'].nunique()} curves across "
        f"{master_df['phase'].nunique()} phases."
    )

    x_interp = np.linspace(0.0, 1.0, INTERPOLATION_POINTS)
    _, mean_curve_df = build_mean_curves(master_df, x_interp)
    prompt_metrics_df = compute_prompt_metrics(master_df)

    phase1_hardware = None
    phase1_prompt = None
    normative_result = None
    pca_result = None
    training_result = None

    if not args.skip_phase1:
        phase1_hardware = run_phase1_hardware_invariance(master_df, mean_curve_df, prompt_metrics_df, plots_dir, tables_dir, x_interp)
        phase1_prompt = run_phase1_prompt_invariance(master_df, mean_curve_df, prompt_metrics_df, plots_dir, tables_dir, x_interp)

    if not args.skip_phase3:
        normative_result = run_phase3_normative_overlay(master_df, mean_curve_df, plots_dir, tables_dir, x_interp, args.min_params_b)
        if not args.skip_pca:
            pca_result = run_phase3_pca(normative_result, plots_dir, pca_dir, x_interp)
        training_result = run_phase3_training_emergence(master_df, prompt_metrics_df, plots_dir, tables_dir)

    if not args.skip_report:
        generate_markdown_report(
            output_dir,
            master_df,
            phase1_hardware,
            phase1_prompt,
            normative_result,
            pca_result,
            training_result,
        )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
