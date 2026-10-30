# Leakage controls for Experiment 02

These additive notebooks audit the evidence that reaches each report generator. They keep the existing BR00 protocol, root Python helpers, configuration and generation paths intact. BR01B extraction can continue; no restart or bank rebuilding is required.

## Upload

Extract `experiment02_leakage_controls_update.zip` **inside**:

```text
/path/to/repository/experiment02
```

Keep `extensions/leakage_controls.py` in its `extensions` subfolder. Do not copy it over `generation.py`, `retrieval.py` or other frozen helpers. The update contains new notebooks/helper/tests/documentation; the original signed root Python files are unchanged.

## Running order

1. Finish BR01B and BR01's final bank-finalization cell, then BR02 and BR03 as before. For Shaoxing, parent notebook 13 and BR02A are still prerequisites.
2. Run **BR03A_audit_retrieval_and_prompt_leakage.ipynb** in the CPU/graph environment. Require both `complete: true` and `passed: true`.
3. Run **BR04A_generate_reports_with_leakage_gate.ipynb**, replacing BR04, once per generator in its established virtual environment. Start with five reports, inspect the pilot, then set `MAX_NEW_REPORTS=0`. Generation repeats the audit, so a stale passing notebook output cannot authorize new prompts.
4. Finish all generators, then run BR05 with the existing folder paths.

For k=1–5 sensitivity, use **BR06A_guarded_k_sensitivity.ipynb** to prepare/audit child runs and generate guarded reports. Use BR06's existing evaluation/aggregation cells afterward; prepare `CONFIGS` there with the same sweep version. BR06A replaces its generation route, not its evaluation. Do not run both generation routes on the same folder simultaneously.

Run audits independently for each configured cohort. Banks remain shared; audit results and reports are cohort-specific. No `config.json` change or new parent `run_dir` is needed just to add this gate.

## Enforced checks

| Boundary | Check |
|---|---|
| Prepared Symile patients | Training, validation and test patient sets are disjoint. |
| Frozen graph | Reconstruct vocabulary, edges and distinct patient/admission support counts from training candidates only; require exact agreement. |
| Symile bank | Training-only paired patient/admission/modality identities; frozen feature spaces, embeddings, sensor row/study IDs and extracted findings. Lab/admission arrays and finalized bundle signatures remain unchanged. |
| MIMIC-CXR bank | Explicit training partition, source-index/patient/study/image agreement, exclusion of every raw Symile patient's ID, one study image, no duplicate bank image hashes, approved sensor-only MedGemma provenance. |
| Forward references | Known training admissions; distinct patients; exclude the internal query patient. |
| Reverse references | Known training admissions; distinct patients; exclude all forward-anchor patients and the internal query patient. Random reverse controls obey the same policy. |
| Query candidates | Frozen held-out sensor-only records internally; attempted sensor-only PULSE extractions externally. |
| Graph evidence | Recompute real/shuffled support, conflict statements and local graph from approved sensor findings. Unknown edges, changed support counts and injected conflict text fail. |
| Selected CXR context | Approved pool IDs, frozen rank fields, requested k and distinct-patient selection. Selected references are never silently replaced. |
| Prompt cases | Reconstruct all five cases from approved bank findings/labs and graph support. Exact comparison rejects extra fields, changed evidence and supplied query reports/labels. |
| Before inference/resume | Check the audited case, packet and retrieval hashes and monitored file metadata again. Abort on change before the native inference/error-catching block. |

The audit checks recorded selection/provenance; it does not repeat the expensive nearest-neighbor searches or independently benchmark their numerical rankings. The original retrieval stage and matrix hashes remain authoritative for scores. Internal queries use the MIMIC patient namespace. Shaoxing identifiers are treated as a separate namespace; numeric coincidences cannot establish cross-dataset patient identity.

The helper does not parse or use independent target labels. The inherited frozen-artifact validator may **hash** target files to confirm integrity. Hashing a frozen target file is not passing its labels into selection or a prompt. Evaluation remains the only stage that uses those independent labels for scoring.

## Audit and generation outputs

With the default run, audit outputs are:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/audits/leakage_controls/<cohort>/summary.json
/path/to/repository/outputs/real_run_001/experiment02_v1/audits/leakage_controls/<cohort>/queries.csv
/path/to/repository/outputs/real_run_001/experiment02_v1/audits/leakage_controls/<cohort>/queries/<query_id>.json
```

Per-query JSON includes patient/admission/study IDs, selected references, reverse paths and evidence hashes. Keep these private; they are not blinded-review files. CSV records pass/fail and errors. A global provenance failure writes a non-passing summary and raises. A query failure is logged without deleting or excluding that query; the full audit cannot pass until it is resolved.

Reports remain at:

```text
<run_dir>/generations/<cohort>/<model>/<nonthinking_or_thinking>_t<tokens>/reports.jsonl
```

Each generator folder also contains `leakage_guard_protocol.json`, `leakage_guard_initial_journal.json` and `leakage_guard_checks.jsonl`. The initial journal snapshot distinguishes older reports from new guarded inference. Existing reports can be checked retrospectively, but are not represented as having been guarded when originally generated. Native generation/evaluation protocols and score calculations are unchanged.

## Interruption and changes

Audits write atomic per-query checkpoints and revoke any previous green summary at the start. Rerunning checks all inputs and queries again rather than trusting saved green checks. STOP, timeout or interruption cannot produce a passing incomplete summary. Remove only the intended `audits/leakage_controls/<cohort>/STOP` file to resume. No source data, packets or banks are modified by the audit.

Guarded generation uses the original durable journal, successful-record skipping and incomplete-final-line recovery. `MAX_RUN_HOURS` covers audit plus generation after model loading. The k sweep shares its outer time budget/STOP across children. Set the budget below allocation time remaining. Signals stop between records; an in-flight inference/global hashing operation is not forcibly interrupted by the helper.

No simultaneous writers to one model folder are allowed. Do not modify bank/retrieval/packet inputs while generating. A changed gate version or audited evidence conflicts with its immutable generation guard protocol and stops the run; investigate and use a separate versioned run rather than deleting provenance files.

## What still requires evidence

These checks establish consistency with the recorded protocol, not complete absence of contamination. They do not establish cross-dataset waveform independence, near-image independence, independence of foundation-model pretraining, frozen encoder adaptation history, clinical ground-truth validity or validation-only human decisions. Existing within-Shaoxing deduplication does not prove independence from MIMIC. Preserve these as limitations unless independently checked.

Select k, graph thresholds, prompts and model settings on validation data. Freeze them before test/external evaluation. Do not use query reports, machine diagnoses or evaluation errors to choose reference cases or rewrite their contexts. Independent stateless generator calls are retained by the existing loaders; the gate does not infer factual correctness from a passing provenance check.

CPU tests use small constructed data and mocked inference. Real checkpoint compatibility and real-data audit passage must still be checked on Biowulf.
