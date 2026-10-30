# Data and environments

## Inputs that must be obtained separately

- Official Symile-MIMIC release: split CSVs, `data_npy` arrays, `labs_means.json`, and patient/admission identifiers.
- Prepared `manifest.json`, `sensor_sources.json`, `embeddings.json`, `candidates.json`, and independent `annotations.json`.
- Original ECG-HuBERT and RAD-DINO feature arrays and row-aligned manifests. Notebook 00c imports these existing assets; the original upstream feature-bank production pipeline is outside this source release.
- Eligible MIMIC-CXR images, a row-aligned manifest, feature/status arrays, and a training-only partition. An all-split manifest is not a substitute for that partition.
- Frozen parent run: `records.json`, `targets.json`, `graph.json`, `freeze.json` and all artifacts/sources named in that freeze.
- External `ecg-arrhythmia/1.0.0`: WFDB records and `ConditionNames_SNOMED-CT.csv`, followed by root 12/12B/13 QC and feature generation.
- Encoder/model checkpoints and upstream PULSE/LLaVA code under their original licenses.

`artifact_root/config.json` must identify the original HuBERT/RAD-DINO models and cache; adapters expect `Edoardo-BS/hubert-ecg-base` and `microsoft/rad-dino`. Manifest paths must point to accessible files. Do not regenerate feature spaces with a different processor or encoder and reuse old hashes.

`lab_code_table.json` contains the lab-column mapping used by the project. It is not patient data. Confirm its column order against the official release before use. Its original source provenance is retained with local machine prefixes replaced by placeholders.

## Environment separation

- CPU review/retrieval/evaluation: Python 3.10+ recommended, root `requirements-cpu.txt`.
- PULSE: the study used a working Python 3.9 environment with torch 2.1.2 and transformers 4.37.2; use the original upstream repository and installation instructions. This release does not provide a complete historical lockfile.
- RAD-DINO/ECG-HuBERT: use the original CUDA environment and preprocessing, then pass numerical equivalence checks.
- MedGemma and modern Llama/Qwen: separate compatible CUDA/Transformers environments. Explicit model paths/revisions, precision, processor and generation settings must be recorded. A single unpinned install is not certified for every model.
- Legacy root 02/03 training: FAISS, torch and torch-geometric from root `requirements.txt`, separate from PULSE. These requirements are legacy development requirements, not the CPU-only Experiment 02 installation.

Use `python -m venv` and register each kernel. Do not upgrade a working signed run's environment mid-run. Save `pip freeze` and checkpoint commit hashes alongside private run artifacts. Python 3.9 is supported by the PULSE helper path; this does not imply every dependency in the newer CPU requirements supports 3.9.

## Configuration

Root `config.example.json` configures preparation; Experiment 02's example config configures BR notebooks. Copy each to its adjacent ignored `config.json` and replace absolute placeholder paths. Set `project_root` to the clone root, not `experiment02`. Choose fresh `output_dir`/`run_dir` values for this public release. Replace explicit path variables in the chosen notebooks before execution.

Tokens belong in a private file outside the repository. Downloaded models and research data must remain outside tracked paths. The MIT license covers project software only; dataset access is governed by PhysioNet and the relevant data-use agreements.
