# Emergent Depthwise Activation Structure In Transformer Language Models

This repository is the code-and-results release for the work on the emergent depthwise activation invariant documented in the paper and write-up linked here:

- https://www.techrxiv.org/users/1012464/articles/1376541-emergent-depthwise-activation-structure-in-transformer-language-models-the-hau-curve-and-its-early-training-attractor-pattern

At this point, the bundled analysis pipeline in [scripts/run_statistical_analysis.py]is the intended reproduction path for the paper-facing results included in this repo. It now runs end-to-end on the bundled result logs and regenerates the main structural analyses from the released artifacts.

## What Is In This Repo

- [app/] contains the Hugging Face Spaces app version of the experiment runner.
- [results/] contains the released markdown result logs across phases 1 through 4.
- [scripts/] contains the analysis and reconstruction scripts.
- [canonical_60_names.txt]lists the canonical 60-model cohort used in the later aggregate analyses.
- [requirements.txt] contains the dependencies needed for both the app workflow and the bundled analysis pipeline.

## Main Reproduction Path

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the bundled analysis pipeline:

```bash
python scripts/run_statistical_analysis.py
```

On the current release bundle, this script is intended to reproduce the paper-relevant analyses from the bundled logs, including:

- phase-1 hardware invariance
- phase-1 prompt-pattern invariance
- the normative tri-phasic activation plot
- the model-level PCA overlay across the 60-model set
- OLMo and Pythia developmental/emergence curves

Generated outputs are written under [scripts/analysis_outputs/], including:

- [analysis_report.md]
- [normative_tri_phasic_activation.png]
- [canonical_pc1_curves_60.png]
- [developmental_curves_OLMo.png]
- [developmental_curves_Pythia.png]

## Experiment Runner

The original experiment runner is still included in [app/run_experiment.py]. It captures per-layer activation norms for paired `dense` and `flat` prompts and writes CSV outputs for a single model run.

Run it directly with:

```bash
python app/run_experiment.py --model_id <huggingface-model>
```

Or launch the Gradio wrapper:

```bash
python app/app.py
```

The prompt inventory used by the app workflow lives in [app/prompts.json]

## Data Layout

The released result logs in [results/] are organized by phase:

- `phase_1`: hardware and prompt-set comparisons
- `phase_2_statements_only`: statement-only comparison runs
- `phase_3`: main-model analyses plus developmental checkpoints
- `phase_4`: smaller-model extension of the cohort

The analysis pipeline walks the result tree recursively and parses the granular CSV blocks embedded in the markdown logs. Across the current bundle, it loads the full 60-model cohort plus the OLMo and Pythia checkpoint series used for the developmental analyses.

## Notes

- The repository is still a cleaned extraction from a larger research codebase, so some historical scripts remain in [scripts/] for traceability.
- The current recommended entrypoint for reproducing the released paper figures/results is [scripts/run_statistical_analysis.py].
- The regenerated plots may not be pixel-identical to the originals, but they should be materially consistent with the released analyses and bundled result logs.

- An earlier version of the analysis pipeline included a default parameter threshold (--min-params-b = 3.0), based on an initial hypothesis that smaller models did not exhibit the same activation structure.

Subsequent analysis showed this assumption was incorrect (the effect is more closely tied to layer depth rather than parameter count), and the filter has been removed.

The current pipeline includes the full canonical 60-model cohort in PCA and aggregate analyses.

-The model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B was initially excluded from the aggregate analyses because its original run predated a correction to the activation preprocessing pipeline (log transformation standardization).  

In the original run, numerical compression effects resulted in near-constant activation values across deeper layers, obscuring the underlying structure.

After applying the corrected logarithmic transformation, the model was reprocessed and included in the Phase 3 results.

Inclusion of this model does not materially change the observed structural patterns or PCA results.

It has since been re-run under the corrected pipeline and included in the current release, under phase 3 results.


