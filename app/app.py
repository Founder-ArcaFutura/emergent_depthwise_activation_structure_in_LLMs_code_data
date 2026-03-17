import gradio as gr
import subprocess
import os
from pathlib import Path
import tempfile

RESULTS_NAME = "results.csv"
GENERATIONS_NAME = "generations.csv"

def run_experiment_ui(model_id: str):
    # 1) run the script and capture stdout/stderr
    # We need to run from the script's directory context
    script_dir = Path(__file__).parent
    proc = subprocess.run(
        ["python", "run_experiment.py", "--model_id", model_id],
        capture_output=True,
        text=True,
        cwd=script_dir  # <-- Set the working directory
    )

    full_log = ""
    full_log += proc.stdout or ""
    if proc.stderr:
        full_log += "\n\n--- STDERR ---\n" + proc.stderr

    # 2) Check for the output files in the script's directory
    default_results_csv = script_dir / RESULTS_NAME
    default_generations_csv = script_dir / GENERATIONS_NAME

    downloadable_results_file = None
    downloadable_generations_file = None

    tmp_dir = Path(tempfile.gettempdir())
    safe_name = model_id.replace("/", "_")

    if default_results_csv.exists():
        # move it into /tmp so Gradio will let us return it
        tmp_results_csv = tmp_dir / f"{safe_name}_results.csv"
        default_results_csv.replace(tmp_results_csv)
        downloadable_results_file = str(tmp_results_csv)

    if default_generations_csv.exists():
        # move it into /tmp so Gradio will let us return it
        tmp_generations_csv = tmp_dir / f"{safe_name}_generations.csv"
        default_generations_csv.replace(tmp_generations_csv)
        downloadable_generations_file = str(tmp_generations_csv)

    # 3) return the log and file paths
    return full_log, downloadable_results_file, downloadable_generations_file


with gr.Blocks() as demo:
    gr.Markdown(
        "## 🧠 Model Psychometry Experiment (v2.2)\n"
        "Run the WLI/Δ analysis on an open-weight model and capture its generated output."
    )

    model_in = gr.Textbox(label="Model ID", value="sshleifer/tiny-gpt2")
    run_btn = gr.Button("▶️ Run Experiment")
    log_out = gr.Textbox(label="Run Output Log", lines=25)

    with gr.Row():
        activations_file_out = gr.File(label="Download Activations CSV")
        generations_file_out = gr.File(label="Download Generations CSV")

    run_btn.click(
        run_experiment_ui,
        inputs=model_in,
        outputs=[log_out, activations_file_out, generations_file_out]
    )

# if you're on HF Spaces leave as is:
demo.launch()
