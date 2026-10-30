# Finish the two missing real-data inputs

Copy notebooks **00d, 00e, 00f, 00g**, plus **candidate_preparation.py** and **sensor_candidate_generation.py**, into `/path/to/repository`. Keep your existing config.json. These notebooks read the configured source paths and default to the known Experiment 4 artifact root. No rerun of 00b/00c is needed if their outputs are current.

| Step | Kernel | Result |
|---|---|---|
| 00d_prepare_independent_annotations | Graph environment | `data/prepared/annotations.json` plus coverage audit |
| 00e_generate_pulse_candidates | Working PULSE Python 3.9.15 environment, GPU | Symile ECG sensor-only reports |
| 00f_generate_medgemma_candidates | Working MedGemma environment, GPU | Symile CXR sensor-only reports |
| 00g_assemble_candidate_exports | Graph environment | `data/prepared/candidates.json` plus coverage audit |
| 01_symile_inputs_and_patient_audit | Graph environment | Final records/targets/audit; must print READY |
| 02–06 | Graph environment | Retrieval, learning and evaluation |

00d and the two model runs are independent and may be completed in any order; 00g requires both full model runs. Run one process per modality/output file.

## 00d: independent reference annotations

Checks enriched Experiment 4 `partition/symile/{split}.jsonl` for `ecg_machine_diagnosis`, then the original `external/ecg/manifest.jsonl` for `machine_diagnosis`. Matches both patient and study IDs. Existing generated reports are not inputs. CXR labels come from 00b. If there are no matching usable ECG findings, provide the native MIMIC-IV-ECG `machine_measurements.csv` (or `.csv.gz`) through `ecg_reference_csv` in config.json or the notebook setting. It must contain subject_id, study_id, and native report_0...report_17 fields.

Partial coverage is allowed but explicit: missing, conflicting, uncertain, historical or unmentioned ECG findings are null, not 0. The notebook writes coverage counts per case. The study must report those denominators; merely creating a target row does not establish a complete reference standard. Native ECG machine interpretations are noisy labels, requiring clinical audit. No assertion of clinical correctness is made by the parser. If all ECG labels are unknown, no combined annotations file is created.

## 00e/00f: actual model inference

The notebooks find an existing local model snapshot using common Hugging Face cache locations, including the Experiment 4 default `/path/to/data/hf_cache`. If ambiguous or absent, set `MODEL_PATH` to the same local checkpoint used in your working model environment. PULSE defaults to `/path/to/data/models/PULSE`; adjust `PULSE_REPO` if your source checkout is elsewhere. Neither notebook upgrades dependencies. Reuse your working environments. Required shared packages include NumPy and Pillow; PULSE rendering also uses Matplotlib.

Start with `MAX_NEW_RECORDS = 10`. Inspect saved images and reports in `outputs/symile_sensor_only_v1/`. Then set `MAX_NEW_RECORDS = 0` and rerun the inference cell to finish all pending records. Set `RETRY_ERRORS = True` to retry failed records. Successful cases resume without regeneration after protocol/sensor identity checks. The expected full count is **11,214 ECG records and 11,214 CXR records** for the cohort imported on Biowulf; actual task counts come from the manifest. Do not treat a 10-case smoke run as a complete cohort.

Generation uses only the released Symile sensor tensors, with native admission ID checks. CXR images are reconstructed from ImageNet-normalized tensors. ECG images use 12 normalized-amplitude leads at 500 Hz; do not interpret normalized amplitudes as mV or use them to establish voltage-based findings. Verify lead order and renderer suitability against source preprocessing. Original full-resolution CXR and calibrated ECG inputs would require another versioned adapter if preferred. Do not mix those input protocols within a run.

The protocol records prompt/model/processor/source-code versions, library versions, per-image checksums, per-record tensor checksums, and failure coverage. For a local checkpoint without a snapshot commit name, model files are hashed once. Failed outputs are retained, not silently dropped. Models may still reflect MIMIC pretraining overlap; this is not resolved by graph-level patient separation.

## 00g: candidate assembly

Requires successful sensor-only output for every selected record. Any missing, failed, mismatched or retrieval-conditioned row prevents a new candidates.json from being written. A small rule parser derived from Experiment 4 extracts concepts/polarity and verbatim local evidence spans. Unrecognized findings are not automatically mapped to negative values. Successful reports with no recognized concepts remain explicit empty candidate lists, counted in the audit. Review this pilot parser on development data before freezing the study; it is not a validated extractor and its vocabulary is limited.

The assembly files are regenerable outputs: reruns can replace annotations.json/candidates.json after successful validation. Use a fresh study directory once test evaluation has begun; changes invalidate frozen study artifacts. GPU inference has not been executed by Codex locally. Tests cover data joins, masking, tensor rendering, candidate assembly and mocked resume behavior, not clinical extraction accuracy or model/GPU compatibility.
