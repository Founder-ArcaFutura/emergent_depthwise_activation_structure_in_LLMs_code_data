import os
import re
import glob
from io import StringIO

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# =====================================================================
# CONFIGURATION
# =====================================================================

PYTHIA_DIR = r"results/phase_3/pythia"
OLMO_DIR    = r"results/phase_3/OLMo"
OUTPUT_DIR  = "hau_single_checkpoint_output"

INTERPOLATION_POINTS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# HELPERS
# =====================================================================

def load_md_file(fp):
    """Extract RAW DATA CSV from one .md file."""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            t = f.read()
    except:
        with open(fp, "r", encoding="latin-1") as f:
            t = f.read()

    m = re.search(
        r"--- RAW GRANULAR DATA.*?---\n(.*?)\n--- END OF RAW DATA ---",
        t,
        re.DOTALL
    )
    if not m:
        print(f"WARNING: No RAW DATA in {fp}")
        return None

    csv_block = m.group(1)
    df = pd.read_csv(StringIO(csv_block))
    return df


def normalize_layers(df):
    max_layer = df["layer_index"].max()
    df["layer_index_norm"] = df["layer_index"] / max_layer if max_layer > 0 else 0
    return df


def interpolate_and_zscore(group, n_points):
    """Z-score one curve and interpolate to fixed resolution."""
    g = group.drop_duplicates(subset=["layer_index_norm"])
    g = g.sort_values("layer_index_norm")

    if len(g) < 2:
        return None

    activ = g["activation_norm"].values
    std = activ.std()
    if std < 1e-9:
        z = np.zeros_like(activ)
    else:
        z = (activ - activ.mean()) / std

    interp = interp1d(
        g["layer_index_norm"].values,
        z,
        kind="linear",
        fill_value="extrapolate"
    )

    x = np.linspace(0, 1, n_points)
    return interp(x)


def extract_mean_curve(df, prompt_type, n_points):
    """Mean of all dense or flat curves for one model."""
    curves = []
    for cid, g in df[df["prompt_type"] == prompt_type].groupby("curve_id"):
        z = interpolate_and_zscore(g, n_points)
        if z is not None:
            curves.append(z)

    if not curves:
        return None

    return np.mean(np.vstack(curves), axis=0)


# =====================================================================
# MAIN EXTRACTION
# =====================================================================

def process_model_dir(model_dir):
    """Load the single model .md file from a directory."""
    files = glob.glob(os.path.join(model_dir, "*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files in {model_dir}")

    # If multiple exist, use the largest (likely the full checkpoint)
    files_sorted = sorted(files, key=lambda x: os.path.getsize(x), reverse=True)
    fp = files_sorted[0]

    print(f"Loading {fp}")
    df = load_md_file(fp)
    if df is None:
        return None

    df = normalize_layers(df)
    # curve_id must exist or be created (older files may not have it)
    if "curve_id" not in df.columns:
        df["curve_id"] = (
            df["model_id"] + "_" +
            df["prompt"].astype(str) + "_" +
            df["example_id"].astype(str)
        )

    return df


# =====================================================================
# PIPELINE
# =====================================================================

def main():
    pythia_df = process_model_dir(PYTHIA_DIR)
    olmo_df   = process_model_dir(OLMO_DIR)

    models = {
        "Pythia": pythia_df,
        "OLMo"  : olmo_df
    }

    results = {}

    for name, df in models.items():
        dense = extract_mean_curve(df, "dense", INTERPOLATION_POINTS)
        flat  = extract_mean_curve(df, "flat", INTERPOLATION_POINTS)

        if dense is None or flat is None:
            print(f"Missing dense/flat for {name}")
            continue

        delta = dense - flat

        results[name] = {
            "dense": dense,
            "flat": flat,
            "delta": delta
        }

    # =================================================================
    # PCA: Delta curves from both models
    # =================================================================
    delta_matrix = []
    model_list = []

    for name, d in results.items():
        delta_matrix.append(d["delta"])
        model_list.append(name)

    delta_matrix = np.vstack(delta_matrix)

    # Center
    delta_centered = delta_matrix - delta_matrix.mean(axis=0, keepdims=True)

    pca = PCA(n_components=1)
    scores = pca.fit_transform(delta_centered)
    pc1 = pca.components_[0]
    pc1 = pc1 / np.linalg.norm(pc1)

    # Save canonical curve
    pd.DataFrame({"pc1": pc1}).to_csv(
        os.path.join(OUTPUT_DIR, "canonical_pc1_from_two_models.csv"),
        index=False
    )

    # Cosine similarity
    sims = []
    for i, name in enumerate(model_list):
        v = delta_centered[i]
        v = v / np.linalg.norm(v)
        sims.append(float(np.dot(v, pc1)))

    metrics = pd.DataFrame({
        "model": model_list,
        "pc1_projection": scores.flatten(),
        "cosine_similarity": sims
    })
    metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

    # =================================================================
    # Combined Plot
    # =================================================================

    x = np.linspace(0, 1, INTERPOLATION_POINTS)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))

    for name, data in results.items():
        ax.plot(x, data["dense"], label=f"{name} Dense", linewidth=2)
        ax.plot(x, data["flat"],  label=f"{name} Flat",  linewidth=2)
        ax.plot(x, data["delta"], label=f"{name} Δ",     linewidth=3)

    ax.plot(x, pc1, label="PC1 (canonical from both models)", linestyle="--", linewidth=4, color="black")

    ax.set_title("Dense / Flat / Δ Curves for Pythia & OLMo (200-point, z-scored)", fontsize=18)
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Z-scored Activation (Relative Units)")
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "combined_hau_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("\n--- DONE ---")
    print(f"Saved combined plot → {out_path}")
    print(f"Saved metrics → {OUTPUT_DIR}/model_metrics.csv")
    print(f"Saved canonical PC1 → {OUTPUT_DIR}/canonical_pc1_from_two_models.csv")


if __name__ == "__main__":
    main()
