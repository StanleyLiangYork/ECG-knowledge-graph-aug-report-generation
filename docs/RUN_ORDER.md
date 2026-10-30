# Notebook execution order

## Foundation at repository root

Use the original artifacts if already available and provenance-compatible. To reconstruct the preparation workflow:

1. `00_protocol_and_setup.ipynb`: root configuration.
2. `00b_prepare_symile_sources.ipynb`: aligned manifest, sensor sources and preparation audit.
3. `00c_import_existing_embeddings.ipynb`: import ECG-HuBERT and RAD-DINO features, with identity checks. This importer does not generate the original Symile/CXR feature banks.
4. `00d_prepare_independent_annotations.ipynb`: independent evaluation targets; do not substitute model outputs as labels.
5. `00e_generate_pulse_candidates.ipynb`: ECG candidate findings in the PULSE kernel.
6. `00f_generate_medgemma_candidates.ipynb`: CXR candidate findings in the MedGemma kernel. Complete coverage/recovery before continuing.
7. `00g_assemble_candidate_exports.ipynb` → `01_symile_inputs_and_patient_audit.ipynb` until READY.
8. Historical foundation route: `02_training_reference_banks.ipynb` → `03_graph_and_reranker_development.ipynb` creates the frozen artifacts consumed by BR00. `04_frozen_test_retrieval.ipynb` is the earlier study's test retrieval, not the internal validation cohort used by this manuscript; retained for provenance, not required to retrain the Experiment 02 selector.

The earlier development notebook contains GNN training. Experiment 02 consumes its frozen co-occurrence graph and input contracts, not its GNN as a cross-modal matcher. The release does not replace the historical freeze contract with an invented shortcut.

## Shared reference banks inside experiment02

1. BR00: edit paths; freeze protocol in a new run directory.
2. BR01: prepare aligned bundles and CXR inventory, but wait on the finalizer.
3. BR01A_original_processor_rad_recheck: use the original processor/runtime to reproduce the stored RAD-DINO features. The other two BR01A notebooks are historical diagnostic alternatives, not three mandatory consecutive steps. Do not lower thresholds to force a pass.
4. BR01B: MedGemma image-only candidate extraction; pilot, then complete coverage.
5. BR01 final cell if all cases succeeded. If failures require recovery, run BR01C instead: its recovered-bank finalizer replaces that final cell.

## Internal validation

Set `cohort="internal_val"` in the base Experiment 02 config.

BR02 → BR03 → BR03A → BR04B → BR05A Section A.

- BR04/BR04A are earlier entry points; BR04B is the context-fit workflow used for the manuscript.
- Start BR04B with PULSE to determine common caps on validation. Save the printed child config and select it for all generators. Do not calibrate independently for each generator.
- Run each of `pulse`, `llama_ecg`, `llama_base`, `qwen38` in its own environment. Five-arm pilots precede `MAX_NEW_REPORTS=0`.
- Evaluate each model separately in BR05A Section A. Sections B/C concern historical rendering reconstruction; they are not required for within-generator scores and must not be used to assert pixel equality after a failed comparison.
- BR06/BR06A are optional validation-only k sensitivity, not mandatory main-study steps.

## External ECG validation

1. Root notebook 12: prepare the external cohort and diagnose duplicate/header issues.
2. Root 12B: freeze the explicitly waveform-QC-filtered cohort; then root 13: verify encoder equivalence and encode that cohort.
3. Set base Experiment 02 cohort to `shaoxing`, pointing `shaoxing_cohort` to the QC version. Keep the shared reference banks/graph fixed.
4. BR02 query preparation cell → BR02A PULSE extraction for every sampled query → BR02 retrieval cell.
5. BR03 → BR03A.
6. BR04B PULSE context-preparation cells: set `FROZEN_VALIDATION_CONFIG` to the internal-validation child config, leaving `CONTEXT_CONFIG_PATH=None` for initial external child preparation. This checks the frozen caps; it does not select new caps from external outcomes.
7. BR04C in the original PULSE rendering environment: archive and verify query images.
8. BR04B per generator using the **external** child config. The public copy calls `generate_archived_context_fit` for Shaoxing. All generators must use the verified archive. Keep `ENABLE_THINKING=False`.
9. BR05S after clinical concept/annotation mapping review. Set `MAPPING_REVIEW_NOTE` to an actual completed review statement. An empty/draft note is not permission to score.

The manuscript external base-Llama analysis uses the later `nonthinking_t1024` run. Preserve but exclude its earlier `nonthinking_t384` run. PULSE and ECG-Instruct retain 384; Qwen retains 1,024. This difference is part of the protocol and must be disclosed.

## Human review (original helper package currently missing)

Do not run this route until `experiment02/clinical_review` is restored from the original study implementation. The available notebooks and import helper are preserved without participant records.

BR07 (internal) or BR07S (external) → BR07A independent-interpretation import → BR08 report release → BR08S rating-form export (external) → BR08A/BR08AS rating import → BR08C adjudication-form export → BR08B/BR08BS/BR08D matching import → BR09 analysis.

Choose the version of each import notebook matching the form schema. Keep one `REVIEW_DIR` and `REVIEW_VERSION` throughout each package. A second model pair uses a new package. Record the reviewer's real expertise/involvement, randomize anonymous report review, and freeze independent interpretations before releasing reports. Same-reviewer adjudication is a second review pass, not inter-rater validation.

## Resumption

Successes are skipped under unchanged signed protocols. Review errors and coverage; `MAX_NEW_REPORTS=0` means no per-call report count limit, not zero work. Increasing output tokens changes the output folder/protocol and is a new run, not an implicit merge. Locks prevent concurrent writers to the same folder. Never unlink an active lock to bypass it. Keep runtime limits below the scheduler allocation remaining.
