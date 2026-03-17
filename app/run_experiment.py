import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from pathlib import Path
import gc
import datetime

# Dictionary to store layer activations
activation_data = {}

# ============================================================
# Hook Factory — Simplified and Guaranteed Monotonic (log1p)
# ============================================================
def create_hook(layer_index):
    def hook_fn(module, input, output):
        global activation_data

        try:
            # 1) Robust Tensor Extraction
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif isinstance(output, dict):
                hidden_states = output.get("hidden_states", next(iter(output.values())))
            else:
                hidden_states = output

            if not isinstance(hidden_states, torch.Tensor):
                activation_data[layer_index] = np.nan
                return

            # 2) L1 norm, forced to float32 before .item()
            raw_norm = hidden_states.float().norm(p=1).item()

            # 3) NUMERICAL STABILITY: Intercept Inf/NaN and 0
            if not np.isfinite(raw_norm) or raw_norm <= 0:
                if not np.isfinite(raw_norm):
                    # Overflow case: treat as max signal for compression
                    stable_norm = 1e308
                else:
                    # Zero/negative case: treat as min signal for compression
                    stable_norm = 1e-12
            else:
                stable_norm = raw_norm

            # 4) MONOTONIC SOFT COMPRESSION
            # Use np.log1p (log(1+x)) for gentle, stable compression.
            final_norm = np.log1p(stable_norm)

            # 5) Store the result (guaranteed finite)
            activation_data[layer_index] = float(final_norm)

        except Exception as e:
            print(f"[Hook error at layer {layer_index}] {type(e).__name__}: {e}")
            activation_data[layer_index] = np.nan

    return hook_fn

# ============================================================
#  Locate MLP layers (cross-architecture)
# ============================================================
def get_mlp_layers(model):
    """Return all MLP (feedforward) submodules in a model."""
    mlp_layers = []
    # GPT-2 / GPT-J style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, block in enumerate(model.transformer.h):
            if hasattr(block, "mlp"):
                mlp_layers.append(block.mlp)
    # OPT / LLaMA / Mistral style
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, block in enumerate(model.model.layers):
            if hasattr(block, "mlp"):
                mlp_layers.append(block.mlp)
    # GPT-NeoX / Pythia style
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for i, block in enumerate(model.gpt_neox.layers):
            if hasattr(block, "mlp"):
                mlp_layers.append(block.mlp)
    else:
        raise TypeError(
            f"Could not find MLP layers for model of type {type(model).__name__}. "
            "Update get_mlp_layers for this architecture."
        )
    if not mlp_layers:
        raise TypeError(
            f"No MLP layers found for model {type(model).__name__}. "
            "Check architecture or submodule names."
        )
    return mlp_layers

# ============================================================
#  Core analysis functions (WLI, WSD, Delta, and Phi)
# ============================================================
def calculate_wli(df_group):
    """Calculates the Weighted Layer Index (Weighted Mean)."""
    df_group = df_group.dropna(subset=["activation_norm"])
    df_group = df_group[df_group["activation_norm"] > 0]
    weighted_sum = np.sum(df_group["layer_index"] * df_group["activation_norm"])
    total_norm = np.sum(df_group["activation_norm"])
    if total_norm == 0:
        return 0.0
    return weighted_sum / total_norm

def calculate_weighted_std(df_group):
    """Calculates the Weighted Standard Deviation (Spread/Focus)."""
    # WLI must be calculated first, as it is the weighted mean.
    wli_value = calculate_wli(df_group)

    df_group = df_group.dropna(subset=["activation_norm"])
    df_group = df_group[df_group["activation_norm"] > 0]

    # We use the already per-prompt normalized norm as the weight.
    normalized_norm = df_group["activation_norm"]

    # Weighted Variance: sum(weight * (value - mean)^2) / sum(weight)
    weighted_sum_sq_diff = np.sum(
        normalized_norm * (df_group["layer_index"] - wli_value) ** 2
    )

    total_weight = np.sum(normalized_norm)

    if total_weight == 0:
        return 0.0

    weighted_variance = weighted_sum_sq_diff / total_weight

    return np.sqrt(weighted_variance)

def generate_analysis_report(df):
    df = df.dropna(subset=["activation_norm"])
    df = df[df["activation_norm"] > 0]

    # ✅ Normalize within each prompt (prevents scale drift)
    df["activation_norm"] = df.groupby(["model_id", "prompt"])["activation_norm"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )

    # WLI and WSD per prompt
    def aggregate_metrics(df_group):
        wli = calculate_wli(df_group)
        wsd = calculate_weighted_std(df_group)
        return pd.Series({'wli': wli, 'wsd': wsd})

    # Group by prompt to get the WLI and WSD for each individual prompt
    metrics_results = (
        df.groupby(["model_id", "prompt", "prompt_type"], group_keys=False)
        .apply(aggregate_metrics)
        .reset_index()
    )

    # Average WLI and WSD per model + type
    final_summary = (
        metrics_results.groupby(["model_id", "prompt_type"])[['wli', 'wsd']]
        .mean()
        .reset_index()
    )

    # Reporting
    report_lines = ["\n--- Model Psychometrics Analysis ---"]
    for model_id in final_summary["model_id"].unique():
        report_lines.append(f"\nModel: {model_id}")
        model_df = final_summary[final_summary["model_id"] == model_id]

        try:
            wli_dense = model_df[model_df["prompt_type"] == "dense"]["wli"].iloc[0]
            wli_flat = model_df[model_df["prompt_type"] == "flat"]["wli"].iloc[0]
            wsd_dense = model_df[model_df["prompt_type"] == "dense"]["wsd"].iloc[0]
            wsd_flat = model_df[model_df["prompt_type"] == "flat"]["wsd"].iloc[0]

            # Primary Metric: Abstractive Tendency (Depth Shift)
            abstractive_tendency = wli_dense - wli_flat
            # Secondary Metric: Focus Tendency (Spread Shift)
            focus_tendency = wsd_flat - wsd_dense

            report_lines.append(f"  - Average WLI for 'dense' prompts (Mean Depth): {wli_dense:.4f}")
            report_lines.append(f"  - Average WLI for 'flat' prompts (Mean Depth):  {wli_flat:.4f}")
            report_lines.append(f"  - Abstractive Tendency (Δ):      {abstractive_tendency:.4f}")
            report_lines.append(f"  - Average WSD for 'dense' prompts (Mean Spread): {wsd_dense:.4f}")
            report_lines.append(f"  - Average WSD for 'flat' prompts (Mean Spread):  {wsd_flat:.4f}")
            report_lines.append(f"  - Focus Tendency (Φ):            {focus_tendency:.4f}")

        except IndexError:
            report_lines.append(
                "  - Could not compute full summary. Missing 'dense' or 'flat' prompts."
            )

    report_lines.append("\n--- Interpretation ---")
    report_lines.append("WLI (Weighted Layer Index) is the mean depth of processing.")
    report_lines.append("**Δ (Abstractive Tendency)** = WLI_dense - WLI_flat. Higher Δ implies deeper processing for abstract tasks.")
    report_lines.append("WSD (Weighted Std Dev) is the spread/focus of processing across layers.")
    report_lines.append("**Φ (Focus Tendency)** = WSD_flat - WSD_dense. **Higher Φ implies flat tasks are more focused** (less spread) than dense tasks.")

    return "\n".join(report_lines)

# ============================================================
#  Main experiment runner
# ============================================================
def run_experiment(model_id: str):
    print(f"Loading model and tokenizer for: {model_id}")
    # --- Safety: quantization only if GPU + not tiny model
    use_quant = torch.cuda.is_available() and "tiny" not in model_id.lower()
    print(f"Quantization enabled: {use_quant}")
    experiment_timestamp = datetime.datetime.now().isoformat()

    if use_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # --- Critical for generation: ensure pad token is set ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("Loading prompts from prompts.json...")
    script_dir = Path(__file__).parent
    with open(script_dir / "prompts.json", "r") as f:
        prompts_data = json.load(f)["prompts"]

    activation_results = []
    generation_results = []
    mlp_layers = get_mlp_layers(model)
    num_layers = len(mlp_layers)
    print(f"Found {num_layers} MLP layers.")

    for item in tqdm(prompts_data, desc="Processing prompts"):
        prompt = item["prompt"]
        prompt_type = item["type"]

        # Register hooks
        hooks = []
        for i, mlp_layer in enumerate(mlp_layers):
            hooks.append(mlp_layer.register_forward_hook(create_hook(i)))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        global activation_data
        activation_data.clear()

        # --- Step 1: Activation Capture (Pre-Generation) ---
        with torch.no_grad():
            # Pass inputs through the model to trigger hooks
            model(**inputs)

        # Hooks have now populated activation_data
        for hook in hooks:
            hook.remove() # Clean up hooks immediately

        print(f"Collected {len(activation_data)} activations for prompt: {prompt[:50]}...")
        for layer_index, norm in activation_data.items():
            activation_results.append({
                "model_id": model_id,
                "experiment_timestamp": experiment_timestamp,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "layer_index": layer_index,
                "activation_norm": norm,
            })

        # --- Step 2: Text Generation ---
        with torch.no_grad():
            # Generate text (max 200 new tokens)
            generated_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        # Decode the generated text, skipping special tokens
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        generation_results.append({
            "model_id": model_id,
            "experiment_timestamp": experiment_timestamp,
            "prompt": prompt,
            "prompt_type": prompt_type,
            "generated_text": generated_text,
        })

        # --- Cleanup ---
        gc.collect()
        torch.cuda.empty_cache()

    df_activations = pd.DataFrame(activation_results)
    df_generations = pd.DataFrame(generation_results)
    # --- Analysis & Reporting (on activations) ---
    print(f"Collected total {len(df_activations)} activation rows.")
    print(f"Collected total {len(df_generations)} generated text outputs.")

    print("\n--- Activation Mean by Deepest Layers (sanity check) ---")
    print(df_activations.groupby("layer_index")["activation_norm"].mean().tail(10))

    print("Running analysis directly on activation data...")
    analysis_report = generate_analysis_report(df_activations)
    print(analysis_report)

    print("\n--- RAW ACTIVATION DATA (Copy and Save as CSV) ---")
    print(df_activations.head(20).to_csv(index=False))
    print("--- END OF ACTIVATION DATA ---")

    print("\n--- RAW GENERATION DATA (Copy and Save as CSV) ---")
    print(df_generations.head(20).to_csv(index=False))
    print("--- END OF GENERATION DATA ---")

    # --- Save results to files ---
    save_dir = Path(__file__).parent
    activations_path = save_dir / "results.csv"
    generations_path = save_dir / "generations.csv"

    df_activations.to_csv(activations_path, index=False)
    df_generations.to_csv(generations_path, index=False)

    print(f"\n✅ Activation results saved to {activations_path}")
    print(f"✅ Generation results saved to {generations_path}")
    print("\nExperiment complete.")

# ============================================================
#  Entrypoint
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Psychometrics Experiment")
    parser.add_argument("--model_id", type=str, required=True, help="e.g. 'mistralai/Mistral-7B-v0.1'")
    args = parser.parse_args()
    run_experiment(args.model_id)