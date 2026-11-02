# Knowledge graph-based multimodal augmentation for electrocardiogram report generation

**Authors:** Zhaohui Liang, Niccolo Marini, Sivaramakrishnan Rajaraman, Zhiyun Xue and Sameer Antani*

Division of Intramural Research (DIR), National Library of Medicine (NLM), National Institutes of Health (NIH), Bethesda, MD, USA

\* Corresponding author.

Companion research code for the manuscript **Knowledge graph-based multimodal augmentation for electrocardiogram report generation**.

## Abstract

### Aims

Retrieval-augmented language models can draft electrocardiogram reports, but added reference material may help or distract and is rarely traceable. We evaluated whether organizing multimodal reference context with a training-derived knowledge graph improves recovery of documented electrocardiogram diagnoses across generators.

### Methods and results

An electrocardiogram-only query retrieved admission-aligned electrocardiogram, chest radiograph and laboratory findings from other patients; a frozen co-occurrence graph guided reference selection and supplied attributed associations. Five prompting arms, from no retrieved context to graph-supported and concept-shuffled context, were tested in four vision-language report generators, two fine-tuned on electrocardiogram data and two general-purpose, on 750 internal and 1,000 external electrocardiograms. Relative to retrieval without the graph, graph augmentation increased an internal finding-recovery score by 0.04 and 0.11 in the general-purpose generators and increased external recall of documented diagnoses in all four generators by 3.9 to 7.2 percentage points, each significant after correction for multiple comparisons. Gains over generation without retrieved context were small for the fine-tuned generators, and the original graph did not outperform the shuffled control. A human reviewer with 11 years of clinical experience assessed 160 reports: clinically significant errors decreased for some generators, omissions and acceptability changed inconsistently, and most reports required modification.

### Conclusion

Knowledge graph-based organization of multimodal reference context improved recovery of documented electrocardiogram diagnoses relative to retrieval alone and made the evidence behind generated statements inspectable; the value of the specific associations and clinical benefit remain unproven. The approach suits drafting support under physician over-read, pending a blinded multi-reader study.

## Code overview

An ECG query retrieves aligned ECG–CXR–laboratory reference admissions from Symile. Their CXR embeddings retrieve MIMIC-CXR references. Reverse retrieval checks consistency with the query ECG, and a training-derived co-occurrence graph supports reference ranking and attributed prompt context. Frozen generators produce ECG reports under five matched prompting arms.

**Release status:** source-code release with CPU tests and sanitized notebook templates. Real-data replication requires separately obtained datasets, encoder features, checkpoint revisions and frozen preparation artifacts. The original `clinical_review/` helpers are included for package preparation, independent interpretation, report ratings, same-reviewer adjudication and analysis. See [reproducibility status](docs/REPRODUCIBILITY.md). No patient data, reports, reviewer records or model weights are included.

## Methods in brief

| Arm | Evidence supplied |
|---|---|
| `sensor_only` | Query ECG image |
| `forward_pool` | Query image plus aligned Symile references and forward-selected CXR findings |
| `bidirectional_pool` | Forward/reverse consistency selects the reference context |
| `bidirectional_graph` | Adds graph-supported selection and attributed association text |
| `bidirectional_shuffled_graph` | Same graph procedure after a fixed semantic permutation within modality/polarity groups |

The graph consists of modality–concept–polarity nodes and undirected co-occurrence edges with distinct-patient/admission support. Present ECG–CXR associations supported by at least two training patients contribute to retrieval. Laboratory percentiles enter reference context; they are not graph nodes or clinical reference ranges. The main Experiment 02 selector is deterministic and **does not train a GNN**. Earlier GraphSAGE/graph-transformer development is retained only as foundation/legacy code where required to reconstruct frozen artifacts.

The principal comparison is `bidirectional_graph` versus `bidirectional_pool`. It measures the combined selection/context contribution. Co-occurrence and reverse similarity do not establish causal relationships. Retrieved observations belong to other patients, not the query patient.

## Generators

| Key | Configured checkpoint | Primary output token limit |
|---|---|---|
| `pulse` | `PULSE-ECG/PULSE-7B` | 384 |
| `llama_ecg` | `convaiinnovations/ECG-Instruct-Llama-3.2-11B-Vision` | 384 |
| `llama_base` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 384 internal; 1,024 external post hoc run |
| `qwen38` | `Qwen/Qwen3.8-27B` | 1,024 |

Primary generation disables thinking. These are the IDs configured in source, not independent proof of the weights loaded in a historical run. Preserve downloaded commit revisions and generation protocols. Each checkpoint has its own access conditions and license.

## Repository layout

- Root notebooks and Python helpers: Symile preparation, independent targets, candidate extraction, legacy frozen banks/graph, and external ECG quality control/encoding.
- `experiment02/BR*.ipynb`: retrieval, guarded generation, automated evaluation and clinical workflow notebooks.
- `experiment02/extensions/`: leakage checks, context fitting, immutable query image archives, extraction recovery and optional k-sensitivity.
- `docs/RUN_ORDER.md`: recommended manuscript workflow and alternatives.
- `docs/DATA_AND_ENVIRONMENTS.md`: required inputs, paths, environments and external assets.
- `docs/REPRODUCIBILITY.md`: missing components, public-copy changes and test scope.
- `docs/source_manifest.json`: original/release file hashes.

Keep this directory structure: Experiment 02 imports the parent helpers. Run root notebooks with the repository root as the working directory; run BR notebooks inside `experiment02`.

## Installation and configuration

Use separate Python virtual environments, not one combined model environment:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m ipykernel install --user --name ecg-kg-cpu --display-name "ECG KG CPU"
cp config.example.json config.json
cp experiment02/config.example.json experiment02/config.json
```

Replace every `/path/to/...` placeholder in configuration and the selected notebooks. Set Experiment 02 `project_root` to the absolute clone root. The root config and `experiment02/config.json` serve different stages. Use an entirely new output directory for this release: sanitized paths and notebook updates change source hashes, so this public copy is not a drop-in resume environment for signed historical runs.

CPU dependencies do not install GPU models. PULSE uses its original LLaVA repository and a separate working environment; newer VLMs require their own compatible Transformers stack. See [environment notes](docs/DATA_AND_ENVIRONMENTS.md). Never publish the Hugging Face token file. Set `TOKEN_FILE` to a private file outside the clone, or use pre-downloaded checkpoints.

## Start here

1. Prepare the foundation artifacts, or supply your authorized existing artifacts with matching provenance.
2. Run BR00, BR01 preparation, the appropriate BR01A equivalence check, BR01B extraction, and bank finalization.
3. For internal validation: BR02 → BR03 → BR03A → BR04B per generator → BR05A Section A per generator.
4. For external validation: root 12/12B/13 first; BR02 preparation → BR02A → BR02 retrieval → BR03 → BR03A → BR04B context preparation → BR04C image archive → BR04B per generator → BR05S.

Detailed cell-level order, recovery variants and clinical review dependencies are in [RUN_ORDER.md](docs/RUN_ORDER.md). The study used 750 internal validation queries and a seeded sample of 1,000 external ECGs; this is not a claim of a new held-out internal test evaluation.

## Evaluation and interpretation

Internal evaluation reports masked finding F1, omission, reference-discordant positives and ROUGE-L where independent narratives exist. External SNOMED CT diagnoses provide positive-only targets: diagnosis recall and omission are interpretable, while F1 can reduce to `2R/(1+R)` and is not an independent endpoint. No external ROUGE-L is computed without real reference narratives. MRR/nDCG require independent relevance judgments.

Report coverage, failures and matched-query denominators alongside quality. Historical rendering differences limit cross-model conclusions; within-generator arm comparisons retain matched inputs. Human review is exploratory single-reviewer calibration with a separate same-reviewer adjudication pass, not independent adjudication or an inter-rater study. Code audits do not establish cross-dataset near-duplicate exclusion or absence of foundation-model pretraining overlap.

## Tests

```bash
python tools/check_release.py
python tools/run_cpu_tests.py
```

Tests use artificial fixtures and mocked model inference. They do not run clinical data or validate checkpoint behavior. Missing clinical-helper tests are reported explicitly, not represented as passing. Optional FAISS acceleration and GPU/legacy training environments require separate validation.

## Data and licensing

Obtain Symile, MIMIC-CXR/MIMIC ECG and the PhysioNet `ecg-arrhythmia/1.0.0` release through their original distribution channels, satisfying credentialing and data-use terms. “Shaoxing” is the study shorthand for the external release; it should not imply all records came from one hospital. No license to those datasets or third-party checkpoints is granted here.

Original project code is provided under the [MIT License](LICENSE). Third-party datasets, model weights, dependencies and upstream PULSE/LLaVA code retain their own licenses. This repository contains research software, not a validated clinical device.

## Citation

The manuscript citation will be added when publication metadata are available. For now cite this repository and the exact commit used. See [CITATION.cff](CITATION.cff) and [data/model references](docs/REFERENCES.md).
