> **Companion release:** follow [the current run order](../docs/RUN_ORDER.md) first. The notes below include historical entry points. Clinical helper availability is documented in [reproducibility status](../docs/REPRODUCIBILITY.md).

# Experiment 02: bidirectional retrieval for ECG reports

This folder implements the extension in `../bidirectional_retrieval_plan.md`. It uses an ECG query to retrieve aligned Symile admissions, expands their CXR evidence into an eligible MIMIC-CXR bank, checks the reverse route back to ECG, and compares graph-supported evidence with four control conditions.

The graph module uses frozen co-occurrence associations. The existing same-modality GNN head is not treated as a trained cross-modal matcher. Laboratory observations are attributed reference context; the existing graph has no lab nodes.

## Upload and first run

Copy this entire folder to:

```text
/path/to/repository/experiment02
```

The existing parent Python helpers and frozen artifacts must remain available. Start Jupyter in `experiment02`. Run BR00 to create **this folder's** `config.json` from the example, edit the paths shown there, and then run its preflight cell. Parent `config.json` is not edited.

Defaults point to your existing Biowulf artifacts. Confirm the MIMIC-CXR training partition and the parent frozen output directory. The lab mapping is included as `lab_code_table.json`, copied from Experiment 4 with its original source provenance. A missing asset yields an explicit error and readiness audit; real data are never replaced with dummy files.

Start with `cohort="internal_val"` for debugging. After freezing the implementation and settings, use `internal_test` and then `shaoxing`. Cohort is a routing selector: it can change within the same experiment run, because each query/retrieval/generation stage has a cohort-specific signed directory. All other configuration is frozen; a changed k, prompt budget, bank or setting needs a new `run_dir`.

## Notebook order

| Order | Notebook | Environment / purpose |
|---|---|---|
| 1 | BR00 | CPU/graph: edit new config, validate frozen parent and input files |
| 2 | BR01, first code cell | CPU/graph: prepare aligned admission bundles and eligible CXR inventory |
| 3 | BR01A | Original RAD-DINO CUDA environment: verify both feature spaces numerically |
| 4 | BR01B | MedGemma CUDA environment: extract image-only findings from eligible CXR references |
| 5 | BR01, final code cell | CPU/graph: finalize normalized modality-specific indexes after extraction completes |
| 6 | BR02, first code cell | CPU/graph: freeze query IDs, raw feature references and label-independent sampling |
| 7 | BR02A, Shaoxing only | PULSE CUDA environment: extract query concepts for the selected sample |
| 8 | BR02, final code cell | CPU/graph: forward/reverse retrieval, graph associations, chance-control diagnostics |
| 9 | BR03 | CPU/graph: build five-arm packets and retrieval diagnostic outputs |
| 10 | BR03A | CPU/graph: audit every retrieval path and prompt packet for leakage |
| 11 | BR04A, once per generator | Generator-specific virtual environment: guarded pilot and resumable report generation |
| 12 | BR05 | CPU/graph: evaluate independent targets and prepare blinded clinical review |

BR01A and BR01B use different GPU environments. Do not keep both models loaded on the same allocation unless capacity permits. Their outputs are checked again when BR01 finalizes banks. You do not need to rebuild the shared banks when routing to the next cohort.

For external queries, finish **parent notebook 13** first. Its outputs must be in `shaoxing_external_waveform_qc_v2/features/`, in the original bank space. The default query sample is 1,000 seeded records; BR02A processes this sample rather than requiring all 44,812 ECGs to undergo candidate generation. Every sampled query candidate must be attempted. Failed candidates produce explicit empty graph support, not query exclusion.

## Requirements

Use Python virtual environments, not conda. The CPU environment needs NumPy, Pillow and rouge-score. `faiss-cpu` is recommended for faster exact inner-product search; `search_backend="auto"` falls back to exact NumPy. `nbformat` is needed only for notebook validation/tests.

Reuse the established PULSE, MedGemma, RAD-DINO, Llama and Qwen environments for their respective notebooks. Their Transformers versions need not be identical. Do not upgrade a working environment needed to resume another signed run. The new association selector does not require torch-geometric; original frozen parent artifacts are validated without retraining.

Model loading follows your working parent adapters. BR04 can optionally download a snapshot with `DOWNLOAD_MODEL=True`; it reads only the first line of `/path/to/private/hf_token` and pins the revision. Model-specific repository access is still required. BR01A/BR01B/BR02A use cached snapshots. Set exact snapshot paths where discovery is ambiguous. Qwen3.8 is a 27B model and requires adequate memory; precision is fixed before the first report.

## What is matched

1. Query ECG → frozen HuBERT space → up to three distinct-patient Symile training admissions, selected from 50 candidates.
2. Each admission supplies its aligned ECG, CXR and observed lab percentiles.
3. Symile CXR → verified RAD-DINO space → 20 eligible MIMIC-CXR candidates per anchor.
4. Each candidate CXR → Symile CXR bank → up to three distinct-patient reverse admissions, excluding forward-anchor patients and the internal query patient.
5. Reverse admissions' ECG embeddings are compared with the query ECG. Forward and reverse ranks are fused with constant 60.
6. Graph arms add unique supported present–present ECG/CXR associations with at least two training patients. Unsupported concepts have zero graph contribution. A local provenance subgraph is retained.
7. Only two distinct-patient MIMIC-CXR references enter the final context. Same initial anchors and candidate pool are used across arms.

The reverse check is consistency evidence; neither it nor co-occurrence establishes causality. Random reverse-link similarities and unrestricted return-to-anchor frequency are saved as chance/trivial-cycle diagnostics.

## Five arms and comparisons

- `sensor_only`: query ECG only.
- `forward_pool`: aligned Symile references and forward-selected CXR references.
- `bidirectional_pool`: references selected using forward/reverse consistency.
- `bidirectional_graph`: consistency plus graph selection and attributed associations.
- `bidirectional_shuffled_graph`: graph arm using a fixed semantic permutation within modality/polarity groups.

Primary contrast: graph minus bidirectional pool. Secondary: bidirectional minus forward; forward minus sensor-only; real graph minus shuffled graph. D–C evaluates the combined graph selection/explanation module, not a separate causal or learned cross-modal effect.

Check the graph-control changed-edge count. Tiny vocabulary groups may not change under permutation, limiting control strength. Graph edge counts are distinct training-patient/admission supports, not probabilities.

## Leakage, alignment and source safeguards

- ECG/CXR admission joins check patient, admission and representative row indices.
- Lab masks and observed percentiles are verified against the official train CSV and column mapping. Missing values stay missing; percentiles are not clinical ranges.
- CXR eligibility uses an explicit Experiment 4 training partition. Every Symile patient's numeric MIMIC subject ID, including patients outside a prepared subset, is excluded.
- Missing/non-numeric subject IDs stop bank construction. Exact duplicate image groups are excluded; one canonical view per study is selected deterministically.
- Forward/reverse spaces are separate. RAD-DINO numerical probes must pass on both Symile and external CXR inputs.
- MedGemma extraction sees only reference images. Query extraction sees only ECG images. Report text and labels in source manifests are not copied to the banks.
- `targets.json` is read only by evaluation. Cross-source waveform/near-image duplicates and model-pretraining overlap are explicitly not established by these checks.

## Generation and interruption recovery

BR04A defaults to five new reports, one query across all arms. Inspect all five, then set `MAX_NEW_REPORTS=0` for all remaining. MedGemma/PULSE extraction notebooks use a three-record pilot. Read coverage and failure logs before proceeding.

Every attempt is appended and fsynced. Successful records are skipped, valid saved records are provenance-checked, and an interrupted final JSONL line is backed up/repaired. `RETRY_ERRORS=True` retries failed cases under unchanged settings. Deterministic truncation usually needs investigation rather than repeated identical retries. CUDA assertion/illegal-memory errors stop the loop; restart the kernel before resuming.

`MAX_RUN_HOURS` starts at the relevant run call. Keep it below remaining allocation time after model loading. Signals and STOP files stop between records; hard kills may repeat unsaved work. These safeguards do not renew Slurm allocations. Different generator directories can run independently; simultaneous writers to one directory are rejected.

STOP locations are visible in code: extraction uses `extraction/STOP`, query candidates use `queries/<cohort>/STOP_candidates`, retrieval uses `retrieval/<cohort>/STOP`, generation/evaluation use their own directory's `STOP`. Remove only the intended STOP file when ready to resume.

## Evaluation and clinical review

BR05 reports all-arm matched quality scores, pairwise-complete comparisons and completion rates for every attempted query. Primary report F1 requires positive evaluable reference labels and weights patients equally. Micro-F1 gets a separate patient-cluster bootstrap. Completion-sensitive F1 assigns failed/unattempted reports zero only where positive reference labels permit F1.

The evaluator produces 2,000-replicate paired patient-bootstrap intervals. Primary graph-minus-bidirectional report-F1 p-values use patient sign flips under a sign-symmetry null assumption, with Holm correction across the selected generator family. This is an explicitly exploratory testing choice; inspect its assumptions before manuscript use. Model interactions compare paired differences in graph gains.

ROUGE-L requires independent report text. It is left unavailable for Shaoxing codes unless actual independent narratives are supplied. MRR@2 and nDCG@2 are computed only if a supplied independent relevance JSON completely judges a query's candidate pool. An empty judgment file does not manufacture relevance. The nDCG ideal respects the two-distinct-patient constraint.

Before interpreting low/high scores, manually audit the finding extractor across model families, negation, refusals and generic reports. The prior base-Llama near-zero results make this particularly necessary. Sparse negatives can make precision/discordance misleading.

For Shaoxing, fill `MAPPING_REVIEW_NOTE` only after reviewing concept mapping and annotation completeness. A draft/blank note stops scoring. This documents a human prerequisite; the code cannot certify the clinical mapping.

The review cell creates a new `PUBLIC` folder containing anonymous images, reports and empty rating forms, plus a separate `PRIVATE` linkage folder. Default: ten jointly complete queries × four models × five arms = 200 review items. Share only PUBLIC, give reviewers independent copies and prespecify adjudication. Existing forms cannot be overwritten. No clinical ratings are generated automatically.

## Outputs

Default Biowulf root:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/
```

Major outputs: `protocol.json`, `stages/`, `audits/leakage.json`, `audits/rad_equivalence.json`, `banks/`, `queries/<cohort>/`, `retrieval/<cohort>/`, `packets/<cohort>/`, `generations/<cohort>/<model>/<mode_budget>/reports.jsonl`, `evaluation/<cohort>/<version>/`, and `review/<cohort>/<version>/`.

Evaluation writes `coverage.json`, `per_report.csv`, `summary.csv`, `paired_comparisons.csv`, `results.json` and resumable checkpoints. A new report file, parser, target mapping or evaluation setup needs a new evaluation version. Query targets and reviewer linkage stay outside generator prompts.

## Validation and practical limits

Run CPU tests from this directory with:

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

Fixtures exercise the actual new retrieval, packet, generation-journal, evaluation and review code with mocked GPU inference. Tests cover patient exclusions, ties, separate embedding spaces, FAISS/exact agreement, broken journal recovery, five arms, missing graph support, partial-analysis denominators, duplicate images and malformed identities. They do not validate real checkpoint compatibility or clinical effectiveness.

Real RAD-DINO equivalence and real model pilots must run on Biowulf. Do not bypass missing inputs/equivalence checks. Most initial cost may be the eligible MIMIC-CXR MedGemma extraction. It is shared across subsequent cohorts and report generators. Start with validation debugging; the already examined internal test set remains exploratory. Freeze settings before inspecting external performance.

## Optional final-reference k sensitivity

After BR02 completes, use `BR06_cxr_reference_k_sensitivity.ipynb` to test one through five final MIMIC-CXR references. It reuses the same pools/scores and creates separate child runs without changing the frozen parent configuration. See [K_SENSITIVITY_README.md](K_SENSITIVITY_README.md) for the shared context-budget policy, run order and common-query evaluation. Explore k on validation before fixing it for test/external evaluation.

## Additional leakage gate

After BR03, run **BR03A_audit_retrieval_and_prompt_leakage.ipynb**, then use **BR04A_generate_reports_with_leakage_gate.ipynb** in place of BR04. For the optional k sweep, use **BR06A_guarded_k_sensitivity.ipynb** for generation, then BR06 for evaluation. See [LEAKAGE_CONTROLS_README.md](LEAKAGE_CONTROLS_README.md) for checks, output paths and limitations. These extensions preserve BR00's signed root Python sources and do not interrupt an existing BR01B extraction.

## Failed BR01B extractions

If every image was attempted but token-limit/repetition errors remain, use **BR01C_recover_failed_cxr_candidates.ipynb** in the MedGemma environment. It preserves the original journal, records retries separately, and provides a provenance-aware finalizer in place of BR01's final cell. See [CXR_RECOVERY_README.md](CXR_RECOVERY_README.md). The updated leakage helper verifies the retry provenance.

## PULSE input-context failures

If BR04A loads PULSE but its graph arms report `PULSE context budget exceeded`, use **BR04B_pulse_context_fit_and_generation.ipynb**. It derives a separate run with common caps fitted to the actual image/conversation/output budget. Use that child config for all compared generators and BR05. See [CONTEXT_FIT_README.md](CONTEXT_FIT_README.md). Do not edit the frozen parent config or mix changed prompts into the original pilot journal.

## Single-clinician assessment and adjudication

If BR05 reports different source ECG hashes despite matching waveform hashes, first use **BR05A_rendering_provenance_and_within_generator_evaluation.ipynb**. See [RENDERING_PROVENANCE_README.md](RENDERING_PROVENANCE_README.md). It supports native within-generator analysis immediately, and combined evaluation only after exact historical PNG reconstruction and decoded-pixel equality. It never edits report provenance or weakens the original gate. Mixed-model clinical review likewise requires verified historical reviewer images; do not bypass its image checks.

After BR05, use the new clinical workflow instead of its older review-preparation cell:

**BR07_prepare_single_reviewer_clinical_study.ipynb → BR08_single_reviewer_assessment_and_adjudication.ipynb → BR09_analyze_single_reviewer_clinical_results.ipynb**.

For independent interpretations completed offline in `PUBLIC/reference_cases_completed.json`, run **BR07A_import_completed_reference_reviews.ipynb** between BR07 and BR08 instead of entering every case manually. Its default path is the existing Biowulf calibration package. It performs the full batch validation and import inside notebook cells, with no script command or additional helper upload. Keep the protected `reference_cases.json` unchanged. After it reports all ten references saved and no reports released, continue to BR08.

Read [clinical_evaluation_plan.md](clinical_evaluation_plan.md). Set the exact BR04B child config in all three notebooks. Start with a ten-ECG validation calibration package; a main clinical sample needs its own version and sample-size rationale. The current default review subset is ECG-Instruct and Qwen, using their shared modern historical images, graph versus bidirectional-pool control; all four models/five arms remain in automated evaluation. The updated BR07 uses `clinical_review/archived_inputs.py` to copy verified historical images without rerendering. See [the archived input instructions](clinical_review/ARCHIVED_INPUTS_README.md). A PULSE review needs its own historical rendering group/package.

One ECG-qualified clinician records independent query interpretations before reports are released, then assesses randomized anonymous reports. Disagreement, errors, omissions, uncertainty or uninterpretable images trigger a separate adjudication pass by the **same clinician**. Initial ratings remain immutable; the clinician may retain a discrepancy, revise with reasons, or mark it unresolved. No automatic clinical ratings or inter-rater reliability claims are produced. Per-item atomic saves permit restoration after a kernel restart. Keep PRIVATE linkage and analysis outputs hidden until assessments are complete.

Outputs are under `<selected_child_run>/clinical_review/<cohort>/<review_version>/`; BR09 writes initial/adjudicated CSV tables, paired patient-bootstrap intervals, coverage and a text findings file in `analysis_v1`. The helper lives in `clinical_review/`, leaving BR00's signed root Python inventory and completed GPU runs unchanged. CPU checks: `python -m unittest test_clinical_review_extension -v` (NumPy, Pillow and nbformat for notebook validation).
