# PULSE context-budget recovery for BR04A

PULSE loaded successfully, but the pilot produced only three successful reports. Both graph arms failed before inference because the expanded multimodal input plus output reserve exceeded the model context limit. `context_tokens=1024` is the optional evidence-text cap; it is not the full model input budget. The full budget includes the base instructions, LLaVA conversation template and visual patch tokens.

Do not increase the model's advertised context, silently truncate graph arms, reduce only their output allowance, or rerun identical failed prompts repeatedly. Do not edit the parent config or signed root helpers. The update creates a separate child run with common text caps chosen to fit PULSE, preserving the original pilot.

## Upload and run

Extract `experiment02_context_fit_update.zip` inside:

```text
/path/to/repository/experiment02
```

Keep the `extensions` directory structure. Run **BR04B_pulse_context_fit_and_generation.ipynb** in your working PULSE environment, with `MODEL_KEY='pulse'` and `CONTEXT_CONFIG_PATH=None`. Leave the parent config at `cohort='internal_val'`.

The notebook measures PULSE's visual-token cost and exact conversation tokenization using one generated query image. It checks every existing case's text against that cost and chooses the largest tested common evidence cap that fits with 384 output tokens and a 32-token safety margin. It allocates smaller per-source/lab/relation caps consistently and requires the graph explanation slot to remain included. It does not use target labels or report-quality results to choose the caps. A failed fit raises an error rather than bypassing the limit.

All query images must have the same dimensions as calibration; the generation wrapper checks this before inference. The original PULSE adapter also checks the actual processed image/input budget for every query. This method assumes the frozen flat-patch processor/checkpoint generates the same number of patches for the same image dimensions. An image-size mismatch stops the call; it is not silently omitted.

The child uses the same frozen banks, graph, query identities, retrieval pool/scores, selected reference IDs and complete evidence cases. Formatting applies the new common caps, so actual prompt exposure can be smaller; generation records included/omitted slots as before. Different model tokenizers may still include different amounts of evidence under the same caps; inspect their exposure audits when interpreting comparisons.

Inspect the five-report pilot. Require five successful arms, then set `MAX_NEW_REPORTS=0`. Successful child reports resume normally, with leakage checks and interruption safeguards. The original 3-success/2-error pilot remains in its original folder and is not reused under changed prompts.

## Paths and other generators

Default child config:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/context_fit/v1/internal_val/config.json
```

Default PULSE reports:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/context_fit/v1/internal_val/generations/internal_val/pulse/nonthinking_t384/reports.jsonl
```

`/path/to/home/...` may be the resolved equivalent on Biowulf; use the notebook's printed path. Calibration detail is in the child `audits/context_fit.json`. Child leakage summaries are in its `audits/leakage_controls/internal_val/` directory.

For each other generator, run BR04B in that generator's established environment, change `MODEL_KEY`, and set `CONTEXT_CONFIG_PATH` to the child config above. This skips calibration and uses the same common caps. Keep `ENABLE_THINKING=False` for the primary comparison. All generators compared in this analysis must use this child run; do not mix original larger-budget results with child results.

For BR05, change **that notebook's `CONFIG_PATH` variable** to the child config path before loading it, then rerun its setup/evaluation cells. Do not change the source `experiment02/config.json` to point to the child or overwrite the original protocol. BR01–BR03 and reference extraction do not need repeating for the validation child: BR04B copies/retags the required immutable stages and regenerates identical evidence packets in the child. The leakage helper validates child source provenance and common caps.

For internal test/Shaoxing, first complete that cohort's source query/retrieval/packet stages as usual. In BR04B keep `CONTEXT_CONFIG_PATH=None`, set `FROZEN_VALIDATION_CONFIG` to the validation child config, and run PULSE preparation. It reuses the fixed caps and checks them on that cohort rather than selecting new ones. Then all generators use the printed cohort-specific child config. A failure must be investigated without tuning settings on external report quality. Optional larger-k reference pools need their own budget verification; do not assume a k=2 fit validates k=5.

## Limits and verification

The update does not alter model context capacity, output-quality criteria or clinical evaluation targets. It does not resolve unsupported clinical assertions made by a generator. Retain independent clinical/finding audits, especially for voltage-based claims from amplitude-normalized images.

CPU tests use mocked inference and token-cost calculations, covering unchanged source protocols/evidence, derived leakage checks, all-arm generation/resume, input-size rejection, impossible budgets and reuse of validation caps on test. The exact LLaVA visual-token calibration and real five-arm pilot must run on Biowulf.
