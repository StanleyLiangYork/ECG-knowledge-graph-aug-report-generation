# Experiment 02: clinical evaluation and manuscript implementation plan

## Objective and current method

Evaluate whether graph-supported multimodal retrieval reduces clinically significant errors in generated ECG reports, compared with bidirectional retrieval without graph support. The intended journal is European Heart Journal – Digital Health. Journal fit does not establish clinical benefit or guarantee acceptance.

The existing implementation remains unchanged: query ECG → frozen ECG-HuBERT embedding → distinct-patient Symile training admissions with aligned ECG/lab/CXR → RAD-DINO CXR expansion into the eligible MIMIC-CXR training bank → reverse retrieval → frozen graph support → generator context. The graph is a training-derived finding co-occurrence graph, not a causal model or the original trained GNN used as a cross-modal matcher. The query's own paired reports, labs and CXR are not reference evidence. The graph contrast combines reference selection and association text; it does not isolate those mechanisms.

Keep the existing five automated arms and four generators. Non-thinking generation is primary; Qwen thinking is optional supplementary analysis. Do not add models to satisfy journal positioning.

## 1. Finish and freeze the computational evaluation

1. Complete internal-validation generation through BR04B using the SAME context-fit child config for every generator. Run BR05 for automated metrics and coverage. Do not use BR05's older ten-case review cell for the new clinical workflow.
2. Choose final retrieval k, common context caps, generator revisions, prompts and output budgets using validation only. Preserve graph, patient exclusions, encoder checks and leakage audits. Document any previous inspection of the internal test set; such analyses remain exploratory.
3. Prepare internal-test retrieval/packets/leakage checks and generation with the frozen validation caps. Run BR05 with the matching child config, not the original large-context pilot config.
4. Complete Shaoxing BR02 → BR02A → BR03 → BR03A → BR04B → BR05 using the existing QC-v2 cohort and notebook-13 embeddings. Use the validation-frozen caps and no external tuning. Complete the clinical diagnosis-code mapping review before automated scoring.
5. Retain every failure and attempt count. Label matched-complete analysis explicitly; never remove generation failures without reporting them. Do not fabricate independent narrative targets for Shaoxing or calculate ROUGE-L against code-derived synthetic text.

Internal_val is appropriate for reviewer calibration and workflow testing, not a new held-out effectiveness claim. Finish each cohort independently; the same new notebooks operate on its exact generation child config.

## 2. Lock a focused human-assessment protocol

Use ONE ECG-qualified clinician. The clinician may be an author; record their specialty, years of ECG interpretation experience and involvement in model development. An MD/PhD alone does not certify ECG expertise. State this single-reviewer limitation in the manuscript.

The human primary endpoint is the proportion of reports containing at least one clinically significant error. The prespecified paired contrast is `bidirectional_graph − bidirectional_pool`; a negative difference favors graph support. The automated primary endpoint in BR05 remains its original patient-averaged finding F1; do not retrospectively relabel it. This is an additional clinical endpoint, declared before reading the clinical ratings.

Secondary clinical endpoints: major omissions, acceptable-unchanged reports (quality=5), and ordinal report quality. Retain initial and adjudicated versions. Automated F1, masked Hamming accuracy, omission rate, reference-discordant positive rate and available ROUGE-L remain complementary proxies. Clinical endpoints do not demonstrate improved patient outcomes. This extension does not add MRR/nDCG without independent retrieval-relevance judgments.

Start with ten validation ECGs for rubric calibration. Do not include those ratings in the main clinical estimate. Before a main sample, document a sample-size justification based on the desired paired major-error difference/precision, expected discordant pairs, and a calibration estimate; a fixed number of report rows is not a power calculation. If only a feasibility sample is affordable, label it exploratory and report uncertainty.

To limit workload, the current default human subset uses ECG-Instruct and Qwen, two arms each. These retain an ECG-tuned/general-purpose comparison while sharing the modern historical image rendering. The original PULSE/Qwen default was changed because their historical pixels differ; this is an input-provenance decision, not selection of whichever model benefits most. Fix the subset before human scoring. Automated comparisons continue to cover all four models and five arms. Example workload: 100 unique patient IDs per cohort × two arms × two models = 400 reports per cohort; internal plus external = 800 initial ratings, plus triggered adjudications. This is a planning example, not an asserted sufficient sample size. Expand the selected model/arm list before preparing a review package if resources permit, using one verified rendering group per package.

Sampling is seeded and label-independent: sample recorded patient IDs uniformly, then select one eligible query uniformly within each selected ID, using queries jointly successful across the selected models/arms. The saved coverage enumerates failed/unattempted and excluded queries. This estimates report quality conditional on successful generation; it is not an intention-to-generate clinical safety rate. External record identifiers do not necessarily prove distinct biological patients. Record that limitation where patient linkage is unavailable. Do not combine an enriched rare-diagnosis challenge set with a prevalence estimate without an appropriate sampling/weighting plan.

## 3. One-reviewer assessment with separate adjudication

### A. Independent ECG interpretation

BR07 creates only anonymized ECG cases for the reviewer. Before any generated report is released, the clinician records an independent ECG interpretation, interpretability and limitations for every sampled case. The reference is a single-expert assessment, not a consensus gold standard. Automated diagnoses and reference-patient evidence are hidden.

Default source access is the exact image used by the generators. The updated BR07 copies these images from the completed BR05A modern historical reconstruction via `clinical_review/archived_inputs.py`; it does not rerender through the current kernel. See `clinical_review/ARCHIVED_INPUTS_README.md`. The images are amplitude-normalized; absolute-voltage claims may not be assessable, and numerical intervals may require calibration. Record those limitations. If reviewing calibrated original waveforms instead, decide and document that policy before preparing the package, use identical access for every item, and distinguish information available to the clinician from information available to generators. The implementation supplies generator images only; any additional calibrated material must be provided through a separately documented blinded process. Never provide another patient's retrieved CXR/lab evidence as query truth.

### B. Initial blinded report assessment

BR08 freezes all independent interpretations before releasing randomized anonymous generated reports. Review each report against its query ECG. Hide model/arm names, automatic targets and scores. Report style and repeated ECGs may partly reveal identity; disclose incomplete blinding. Avoid side-by-side arm comparisons.

Enter explicit counts for major errors, major omissions and minor errors; overall quality 1–5; image interpretability; and agreement (`agree`, `disagree`, `uncertain`). Define major as an incorrect/missing finding potentially affecting clinical interpretation or management. Calibration should cover rhythm, AV/intraventricular conduction, QT, ST-T/ischemic assertions, ectopy, unsupported voltage claims, and transfer of another patient's findings. No clinical judgment is filled automatically.

### C. Discrepancy-triggered adjudication

Interpret the requested “reviewer is not consistent with the report” as clinical disagreement/uncertainty, not a requirement to make human and model agree. A separate pass is required for ANY disagreement, uncertainty, nonzero error or omission count, quality≤3, or uninterpretable ECG. Freeze initial ratings first.

The SAME clinician reopens the ECG and independent interpretation in a later session where feasible. Record evidence reviewed, a clinical rationale, and one decision:

- `retain_initial`: confirm the discrepancy; the report remains erroneous if the evidence supports that judgment.
- `revise_initial`: explicitly correct the initial assessment, preserving the original.
- `unresolved`: retain uncertainty without assigning an invented final score.

A model vote, automatic label or pressure to favor graph generation is not adjudication evidence. An initially mistaken ECG interpretation can be acknowledged in the adjudication rationale; the original interpretation remains preserved. Single-reviewer re-review cannot establish inter-rater reliability or independent validation. Do not report kappa/ICC between the initial and adjudicated scores as inter-rater agreement.

## 4. New implementation and running order

| Notebook | Action | Completion gate |
|---|---|---|
| BR07_prepare_single_reviewer_clinical_study.ipynb | Choose exact child config, selected generators/arms, stage, reviewer and sample size; freeze anonymous cases | Reviewer completes an independent interpretation for every sampled ECG |
| BR08_single_reviewer_assessment_and_adjudication.ipynb | Release reports; save one initial rating per report; freeze initial scores; prepare and complete adjudication queue | Every initial rating exists and every triggered item has an adjudication decision |
| BR09_analyze_single_reviewer_clinical_results.ipynb | Validate signed sources; export initial/adjudicated clinical results and paired patient bootstrap intervals | Missing adjudications stop analysis; unresolved decisions remain explicit |

Run after BR05 automated evaluation, first for calibration on internal_val, then a fresh main package on each evaluation cohort. BR07–BR09 need CPU Python with NumPy and Pillow; no GPU/model loading, no torch-geometric, and no conda. Jupyter/IPython displays ECG images. Forms are entered via explicit per-item save calls in BR07/BR08; do not run an unanswered rating cell as if it were a completed assessment.

The notebooks default to the known Biowulf validation child config:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/context_fit/v1/internal_val/config.json
```

For test or Shaoxing, select THAT cohort's actual printed BR04B child config. Do not change a frozen run's protocol or guess that a validation path contains external generation. BR07 prints selected source folders and cohort before preparation. Model token budgets select folders; set them to the actual completed run (default PULSE384, Qwen1024). Do not mix a recovered/new-budget folder with an original folder without documenting the choice before sampling.

All extensions are in `clinical_review/workflow.py`. New notebooks/Markdown and this subdirectory do not change BR00's frozen root `*.py` inventory. Existing retrieval/generation helpers and configurations are untouched. The new review helper itself is hashed into each review protocol: preserve it once collection begins. Do not “fix” an active review by editing submitted JSON or changing the helper; retain artifacts and create a corrected version with documented reasons.

## 5. Interruption and restoration

Each completed case interpretation and report rating is atomically saved in its own JSON file. Restart the CPU kernel, select the SAME `REVIEW_DIR`, inspect `progress()`, and continue with the next unassessed item. Do not rerun package preparation over an existing directory. Initial/reference submissions and adjudications are immutable. A hard interruption can require re-entering the current unsaved rating, but previous saved items remain. This workflow assumes one writer; do not edit the same review directory concurrently. Keep an ordinary backup of human-entered data.

The final CPU analysis is inexpensive. Rerun with unchanged input files and parameters after interruption; it rebuilds derived CSV outputs while preserving immutable analysis provenance. It does not rerun generators. No 36-hour GPU allocation is needed.

## 6. Outputs and statistics

Default review location under the selected child run:

```text
<child_run>/clinical_review/<cohort>/calibration_v1/
  PUBLIC/reference_cases.json                 # anonymous query-only cases
  PUBLIC/reference_drafts/<case_id>.json       # saved clinician interpretations
  PUBLIC/reports.json                         # released only after interpretations freeze
  PUBLIC/rating_drafts/<review_id>.json        # saved initial ratings
  PUBLIC/adjudication_queue.json
  PUBLIC/adjudications/<review_id>.json
  PRIVATE/protocol.json, coverage.json, linkage.json, assets.json
  PRIVATE/references_submitted.json, initial_submitted.json
  analysis_v1/per_report.csv, summary.csv, paired_comparisons.csv
  analysis_v1/results.json, findings.txt, protocol.json
```

Share only PUBLIC with the clinician. PRIVATE and analysis exports identify models/arms/patient IDs and must remain hidden until all ratings/adjudication are submitted. Instructions in PUBLIC do not contain patient/model linkage. Same-reviewer adjudication uses the original rating and ECG; it does not unblind experimental arms.

Report initial and adjudicated major-error proportions, omissions, quality, acceptability, denominators and unresolved/uninterpretable counts. The primary paired difference is graph minus bidirectional pool, computed within query and averaged within patient ID. Bootstrap patient-level differences (2,000 replicates by default). Report cohorts and models separately; no automatic pooling or efficacy declaration. Confidence intervals reflect case sampling, not variation between reviewers. This implementation reports descriptive intervals, not multiplicity-adjusted confirmatory significance tests. Prespecify a separate multiplicity plan before making confirmatory claims across models/endpoints.

Unresolved, still-uncertain or uninterpretable assessments do not become correct reports. Exclude them explicitly from evaluable paired estimates, report their counts, and show simple best/worst bounds for the major-error proportion across all sampled reports. Generation failure counts are a separate computational coverage outcome. Report how many assessments retained or changed the initial judgment and why; same-reviewer revisions are not inter-rater agreement.

## 7. Manuscript deliverables

1. Cohort flow diagram: source patients/records, QC/exclusions, retrieval eligibility, generation completion, clinical sampling, adjudication and unresolved cases.
2. Methods figure: ECG→aligned Symile bridge→CXR expansion/reverse matching→graph support→generation, with a separate Shaoxing evaluation route.
3. Automated results for all prespecified models/arms and clinical paired results for the declared review subset.
4. Error taxonomy with anonymized examples showing both graph benefit and graph-induced unsupported assertions; no selective success-only examples.
5. Limitations: noisy/incomplete machine reference labels, single-reviewer bias, normalized ECG images, unknown pretraining overlap, unestablished cross-dataset waveform independence, and co-occurrence rather than causality.
6. Data/code availability, institutional determination for the secondary deidentified-data review, author roles and AI-use disclosure as applicable.

Relevant journal context: [EHJ-DH author instructions](https://academic.oup.com/ehjdh/pages/general-instructions) and the published [ECG-GPT development/multinational-validation study](https://academic.oup.com/ehjdh/article/7/3/ztag031/8489956). These motivate clinical evaluation; the sample sizes and one-reviewer procedure above are this project's design choices, not journal-mandated rules.
