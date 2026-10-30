# Bidirectional retrieval and graph-supported ECG report generation

**Plan date:** October 15, 2026
**Status:** Proposed extension; implementation and validation remain to be completed.
**Local project:** `/path/to/repository`
**Biowulf project:** `/path/to/repository`

## 1. Objective and claims

Test whether an ECG-only query benefits from (1) reference evidence retrieved through aligned Symile admissions, (2) an explicit reverse retrieval check, and (3) graph-supported selection and organization of that evidence.

The final input remains the query ECG. Retrieved laboratory measurements and CXR findings describe reference patients. They must never be presented as measurements or diagnoses established for the query patient.

Bidirectional retrieval measures consistency, not causality. A successful return path can arise from embedding geometry, common findings or chance. The experiment must measure added value against forward retrieval and appropriate controls; it must not treat cycle completion as independent clinical validation.

**Primary hypothesis:** adding the frozen graph inference module to bidirectional retrieval improves reference-label agreement in generated ECG reports relative to bidirectional retrieval alone.

**Secondary hypotheses:** bidirectional selection improves over forward-only selection; effects differ across generators; useful effects persist on an external ECG cohort. Null and negative results remain reportable.

## 2. Existing assets and constraints

Reuse these assets without modifying their signed files:

| Asset | Current location or role |
|---|---|
| Frozen graph, rerankers, records and targets | `outputs/real_run_001/` under the Biowulf project |
| Existing report experiment | `outputs/real_run_001/report_study_v1/` |
| Original encoder configuration | `/path/to/data/symile/ecg_cxr_bridge_artifacts/config.json` |
| Symile arrays | `/path/to/data/symile/physionet.org/files/symile-mimic/1.0.0/data_npy/` |
| Cached ECG/CXR embeddings | `/path/to/data/symile/ecg_cxr_bridge_artifacts/features/<split>/` |
| Existing MIMIC-CXR artifact candidates | `/path/to/data/symile/ecg_cxr_bridge_artifacts/external/cxr/` |
| Clean external ECG cohort | `outputs/real_run_001/shaoxing_external_waveform_qc_v2/` |

The MIMIC-CXR manifest, embedding array, status array, patient partition and extraction outputs must be audited before reuse. Paths alone do not establish eligible training membership, encoder equivalence or reliable findings.

Code inspection establishes these limitations:

- `core.build_graph` represents ECG/CXR candidate concepts with polarity and same-admission co-occurrence support. It does not encode causation.
- `learning.pair_data` trains the ranking head on same-modality pairs. Do not feed ECG–CXR pairs into that head and interpret its output as validated compatibility.
- The current graph has no laboratory nodes. Labs may be attributed reference context in this extension, but cannot be described as part of learned laboratory reasoning.
- Existing generation and evaluation helpers assume three arms. The new experiment needs separate arm-aware helpers; do not silently map the new arms onto old names.
- Earlier internal test results have already been examined. This extension is exploratory on that test set. Reserve untouched external evaluation for the protocol frozen before inspecting its performance.

## 3. Separate experiment branch

Create code and notebooks in `graph_reasoning/bidirectional_retrieval/`. Write outputs to:

```text
/path/to/repository/outputs/real_run_001/bidirectional_retrieval_v1/
    protocol.json
    audits/
    banks/
    retrieval/<cohort>/
    packets/<cohort>/
    generations/<cohort>/<generator>/<mode_budget>/
    evaluation/<cohort>/
    review/<cohort>/
```

Record checksums for the parent graph freeze, model checkpoints, manifests, feature protocols, candidate extractors, lab mapping and new code. Do not overwrite `config.json`, previous report runs, frozen cohorts or trained weights. A changed protocol, checkpoint, input pool or prompt requires a new run version.

## 4. Input contracts and alignment

### 4.1 Query ECG

For Shaoxing, use the 44,812 retained records from the completed waveform-quality cohort, subject to successful completion of encoding. For internal evaluation, use the existing held-out Symile ECG cohort. Preserve all attempted query IDs in the experiment denominator.

Use the verified frozen HuBERT encoder and the exact preprocessing used for the reference bank. Keep its numerical equivalence check. Never compare HuBERT and RAD-DINO vectors directly.

Use the existing PULSE sensor-only candidate extractor for query concepts across all report generators. Candidate extraction must see only the query ECG, not targets, retrieved reports or laboratory data. Treat extracted concepts as model proposals, not ground truth. If extraction fails, record the failure and graph-query coverage; do not manufacture concepts.

### 4.2 Symile admission bank

Build one record per eligible training admission, with a stable admission ID and patient ID linking:

- ECG embedding, sensor row index and candidate observations;
- aligned CXR embedding, image/study ID and candidate observations;
- observed lab mask and values/percentiles with explicit item IDs and provenance.

Validate patient, admission, split and row-index agreement using the original arrays and manifests. Do not join by row order alone or nearest acquisition time when an explicit admission link exists. Admission alignment does not imply simultaneous acquisition; retain timestamps when available and report missing temporal information.

Use the ECG and CXR from the same verified representative admission row. If several studies are eligible, apply a documented deterministic selection rule established before evaluation.

Preserve missing labs as missing. Symile training-distribution percentiles are not laboratory reference ranges and must not be labeled clinically high/low without supporting raw values and validated ranges. Use item IDs if lab-name mapping is uncertain.

### 4.3 MIMIC-CXR bank

Each eligible entry needs a stable image/study ID, MIMIC patient ID, RAD-DINO embedding, image provenance and CXR candidate findings. Use the frozen MedGemma sensor-only extraction protocol for the primary experiment. Do not mix report-derived annotations and image-derived candidates without a separate source-labeled arm.

Reuse cached embeddings only after checkpoint, preprocessing, pooling, row identity and a numerical probe check agree with the Symile RAD-DINO space. If that check fails, encode both sides consistently in a new versioned bank; do not compare incompatible vectors.

## 5. Leakage and duplicate controls

Complete these before indexing:

1. Symile forward and reverse banks contain training patients only. Exclude the query patient and every held-out Symile patient from eligible reference sources.
2. Crosswalk MIMIC patient identifiers across the Symile and MIMIC-CXR manifests. Exclude **all Symile patients** from the MIMIC-CXR expansion bank where identifiers permit. This stricter rule avoids returning a seed's own CXR or another study from its patient.
3. Remove exact duplicate images/files and duplicate study entries using a fixed audited policy. Allow at most one final CXR reference per MIMIC patient.
4. Retain the completed Shaoxing header, duplicate and waveform-quality audits. Those audits do not establish cross-dataset patient independence.
5. Document coverage of cross-source identity and waveform/image duplicate checks. Missing identifiers, re-encoded signals, near-duplicates and unknown model-pretraining overlap remain limitations; never label an unperformed check as passed.
6. Keep query targets and reference diagnosis text outside retrieval and prompting. Only the evaluation stage may read query targets.

Write input counts, exclusions by reason, unresolved identity coverage and final bank sizes to `audits/leakage.json`. Stop if the required MIMIC patient crosswalk is missing; do not quietly include a patient's own images.

## 6. Retrieval algorithm

Use L2-normalized embeddings and cosine similarity, with separate indexes for ECG and CXR. FAISS inner-product search is suitable; exact matrix search can validate a small pilot. Stable record IDs resolve score ties. Cache repeated searches for shared Symile anchors.

### 6.1 Forward ECG retrieval

For query embedding `e(q)`, retrieve a pool of 50 eligible Symile ECGs. Collapse by patient and select the top **k = 3** admissions; use `k = 2` only as a prespecified sensitivity analysis if needed. Each selected admission supplies its ECG–lab–CXR bundle.

The initial ECG anchors are identical in all retrieval arms. Do not use graph ranking to alter the initial anchors in the primary experiment, because that would confound the later graph comparison.

### 6.2 CXR expansion

For each of the k aligned Symile CXR embeddings, retrieve 20 eligible MIMIC-CXR candidates. Merge by study and patient into one candidate pool, preserving every retrieval path and its cosine score.

For candidate `c`, define forward CXR evidence as:

```text
forward_cxr(c) = max cosine(Symile_CXR(anchor), MIMIC_CXR(c)) over the k anchors
```

Use the maximum rather than summing repeated hits as though they were independent patients. Retain multiplicity as a descriptive audit field.

The initial candidate pool may contain up to 60 studies before deduplication. **The final distinct-patient MIMIC-CXR reference count is a sensitivity parameter, `k_CXR ∈ {1,2,3,4,5}`.** The original default is two. Keep the initial candidate pool fixed while changing only final selection count. Choose a primary value using validation data before test/external evaluation.

### 6.3 Explicit reverse retrieval

For each candidate MIMIC-CXR, search the Symile training CXR bank and obtain **m = 3** reverse admissions from distinct patients. Access the ECG embeddings aligned with those admissions.

For the primary reverse score, exclude the original forward-anchor patients from the reverse search. Otherwise a nearby CXR can simply return the seed that retrieved it. Record the unrestricted return-to-anchor result separately as a diagnostic, not as independent support.

Define:

```text
reverse_ecg(c) = mean cosine(query_ECG, ECG(reverse_admission)) across the m reverse admissions
```

Save every reverse admission ID, both similarity scores and the exclusion decisions. If fewer than m eligible admissions remain, record the actual count; an empty reverse set gives no reverse support. Do not substitute a query label or target report.

### 6.4 Bidirectional selection

Use a fixed rank-fusion rule to avoid treating unlike raw similarities as calibrated probabilities:

```text
bidirectional_score(c) = 1 / (60 + rank_forward_cxr(c))
                       + 1 / (60 + rank_reverse_ecg(c))
```

Ranks are descending, one-based, over the same deduplicated candidate pool; absent reverse support contributes zero. Select the top `k_CXR` candidates from distinct patients. Keep the Symile anchor count separate from `k_CXR`. The constant 60, Symile anchor count, m and candidate limits are fixed defaults for development. Any adjustment must use validation data and be frozen before test generation.

Do not impose a reverse threshold based on test performance. Analyze weak/no-support cases separately. A bidirectional score is retrieval consistency, not a probability of disease.

## 7. Frozen graph inference

### 7.1 First implementation: supported associations

Use the trained graph vocabulary and its frozen co-occurrence edges to construct a local evidence subgraph. Nodes carry concept, modality, polarity and source record. Add explicit provenance edges for admission alignment and retrieval; keep these separate from learned/co-occurrence edges.

Allowed graph evidence includes:

- query ECG candidate concepts connected to reference CXR concepts;
- ECG and CXR concepts from the selected Symile admissions;
- direct co-occurrence edges with their distinct-patient and admission support;
- conflicts between present/absent mentions, without suppressing either source.

Labs remain attributed contextual observations, outside graph scoring in this version. No new disease labels or causal predicates are invented.

For a concrete graph ranking rule, count unique **present–present** query-ECG/reference-CXR edges with at least two supporting training patients. Rank candidates by this count; use the sum of `log1p(patient_count)` as a tie-break, followed by stable ID. No supporting edge gives zero graph contribution. Polarity conflicts are retained in the evidence output and do not create positive support.

Combine this graph rank with the two bidirectional ranks using the same reciprocal-rank-fusion constant. This is a transparent **frozen graph association selector**, not a newly trained cross-modal predictor. Frequent concepts can dominate, so report support distributions and include the graph control below.

### 7.2 Scope of the learned GNN

The existing GraphSAGE/graph-transformer ranking head may be used only for documented same-modality scoring or a separately labeled transfer experiment. It is not the primary cross-modal matcher in this plan.

A learned ECG–lab–CXR matcher is a later extension requiring lab nodes, cross-modal training examples, suitable hard negatives and validation. Do not add it to the minimum experiment or tune it on Shaoxing.

### 7.3 Evidence output

For each query, save:

```text
query_id
query_candidate_observations: [concept, polarity, extractor provenance]
symile_anchors: [patient/admission IDs, ECG/CXR IDs, similarity, labs with missingness]
cxr_candidates: [study/patient IDs, forward paths, reverse admissions, scores]
selected_references_by_arm
supported_graph_edges: [source concept, target concept, patient/admission support]
conflicts_and_unsupported_concepts
selection_rule_version
```

This structured output is the inference result. No additional LLM debate stage is required for the first experiment.

## 8. Report generation arms

| Arm | Evidence selection and context | Purpose |
|---|---|---|
| A: `sensor_only` | Query ECG, no reference evidence | Baseline |
| B: `forward_pool` | Same k Symile anchors plus k_CXR forward-selected MIMIC-CXR references | Value of forward multimodal evidence |
| C: `bidirectional_pool` | Same anchors plus k_CXR references selected with the reverse check | Incremental value of bidirectional selection |
| D: `bidirectional_graph` | Same initial pool, graph-augmented selection, and attributed graph evidence | Incremental value of the graph inference module |
| E: `bidirectional_shuffled_graph` | Same as D with a fixed semantically shuffled graph | Control before claiming meaningful graph structure |

**Primary contrast:** D minus C. **Secondary contrasts:** C minus B and B minus A. D minus E tests whether graph semantics contribute beyond adding a graph-shaped context. D minus C includes both graph-based selection and graph explanation; it does not isolate those mechanisms individually. If warranted later, add a fixed-reference explanation-only ablation.

For E, permute concept identities within modality and polarity using a fixed seed, preserving topology and edge-count distribution. Sign the permuted graph. Its edges are an experimental control and must never be used clinically.

B–E must use the same initial candidates, maximum source count and per-generator total context-token cap. Use a deterministic formatter with reserved slots for source findings, labs and relationship/uncertainty information. Record included and omitted fields and actual token counts. Do not pad with invented information merely to equalize length; analyze residual context-length differences.

The prompt must state that all retrieved evidence concerns other patients and cannot establish query CXR/lab findings. Require a concise ECG report and uncertainty where appropriate. Supply the query ECG image; use structured text descriptions of reference images initially. Adding reference images directly is a separate intervention.

## 9. Generator settings and execution scope

Use PULSE, ECG-Instruct Llama, general-purpose Llama-3.2-11B-Vision-Instruct and Qwen3.8-27B. Keep each model's checkpoint, processor, ECG rendering, precision and decoding fixed across arms. Llama has no equivalent Qwen thinking switch.

Primary Qwen mode is `ENABLE_THINKING=False`. Thinking-enabled Qwen is supplementary; compare both modes at the same total generation-token budget if making a mode-effect claim. Keep budget-matched and unequal-budget comparisons clearly separate.

Do not silently reuse old sensor-only results under a changed prompt, cohort, budget or image format. Reuse requires an exact protocol match. Preserve original truncation/repetition failures in coverage statistics.

Avoid immediately launching four models × five arms × 44,812 ECGs, which would produce **896,240 reports**. First:

1. Use a small fixed validation pilot for functional debugging, with no outcome-guided prompt selection on test cases.
2. Complete the frozen internal held-out comparison.
3. Select a seeded, label-independent external sample of **1,000 eligible ECGs** before inspecting external report performance; this is an operational starting point, not a power justification.
4. Use validation variability and the prespecified meaningful effect size to assess whether a larger sample is needed before starting external evaluation. A larger cohort is optional, not a prerequisite for a defensible manuscript.

If patient linkage is unavailable, label Shaoxing units as records/proxy patients and disclose the limitation. Deduplication does not prove one record per patient.

## 10. Evaluation

### 10.1 Clinical finding agreement

Use independent reference diagnoses/annotations, never generator candidates as ground truth. Freeze a reviewed mapping to the shared concept vocabulary before external scoring. Missing labels remain unknown unless annotation completeness justifies negatives.

Prespecify **patient-averaged report-level F1 on cases with evaluable positive reference labels** as the primary endpoint. Report micro-F1 as a complementary endpoint with its own uncertainty estimate; do not attach report-level F1 intervals to micro-F1 differences.

Also report recall, Hamming accuracy on known labels, omission rate, reference-discordant positive rate, contradiction counts and unassessed positive assertions. Include every metric's denominator. Sparse negative labels make high precision or low discordance weak evidence of factual accuracy.

Before full scoring, manually audit extraction on a fixed blinded sample spanning all models and arms, including base Llama's previously near-zero scores. Check refusals, generic text, synonym coverage, negation, copied references and contradictions. Freeze any parser corrections and rescore all comparable arms consistently. Parser development on test outputs makes those results exploratory.

### 10.2 Text quality and clinical review

Compute ROUGE-L only where independent human/reference report text exists. Shaoxing diagnosis codes are not narrative reference reports; do not create template reports and present ROUGE against them as report-quality validation.

Use blinded ECG-qualified reviewers to rate clinically significant incorrect assertions, omissions and overall report quality. Prepare identical sampled queries across arms/models, randomized report IDs, private linkage and explicit reviewer instructions. Prespecify independent review and adjudication. Record image interpretability and reviewer qualifications.

### 10.3 Retrieval quality

Measure eligibility, duplicate removal, reverse-support coverage, unique reference patients, retrieval latency and source attribution completeness for all queries.

Calculate MRR/nDCG only if independently defined relevance judgments exist. Nearest-neighbor rank or successful return paths are not ground-truth relevance. Sample candidate pools for blinded clinical relevance review where practical. Use a random/shuffled reverse-link control to quantify how much consistency can occur by chance; do not use its results to retune test thresholds.

### 10.4 Missingness, failures and uncertainty

- Report generation completion and failure reasons by model and arm for all attempted queries.
- Primary paired quality analysis uses matched completed reports, with exclusions shown explicitly.
- Add a labeled sensitivity endpoint: assign a failed report F1=0 only for queries whose reference labels otherwise permit the prespecified F1; retain those failures in the denominator. Do not assign zero to unlabeled queries.
- Evaluate D versus C on their full pairwise-complete set, in addition to the all-arm intersection, so failures in A do not unnecessarily remove comparisons.
- Use 2,000 paired bootstrap replicates, resampling patient clusters where identities are available. State whether the estimand weights patients or reports; use patient weighting for the primary report-F1 endpoint.
- Designate the four generator-specific D–C tests as the primary comparison family and apply a prespecified multiplicity procedure such as Holm correction. Treat other contrasts and model interactions as secondary/exploratory.
- Estimate model differences through paired differences in D–C effects, not by comparing which separate confidence intervals exclude zero.

## 11. Proposed implementation notebooks

These names are reserved for new files under `bidirectional_retrieval/`; they do not replace existing notebooks 12–17.

| Notebook | Work | Required outputs / completion gate |
|---|---|---|
| `BR00_freeze_protocol_and_audit_inputs.ipynb` | Resolve files, inspect source/label separation, freeze parameters and cohorts | Signed protocol; leakage audit; explicit unresolved limitations |
| `BR01_prepare_aligned_reference_banks.ipynb` | Build verified admission bundles, audit labs and encoder equivalence, prepare eligible ECG/CXR indexes | Indexed banks, ID-to-row maps, encoder and partition hashes |
| `BR02_forward_and_reverse_retrieval.ipynb` | ECG anchors, CXR expansion, reverse search and rank fusion | Per-query retrieval JSON; complete or explicit failure status for every query |
| `BR03_graph_inference_and_packets.ipynb` | Frozen graph subgraphs, conflicts, graph control, arm-specific selection and formatting | Candidate pool, local graph, prompt packets, provenance and budget audits |
| `BR04_generate_reports.ipynb` | Select one generator per appropriate virtual environment; pilot then resumable run | Durable report journal, protocol, token usage, diagnostics and coverage |
| `BR05_evaluate_and_prepare_review.ipynb` | Independent labels, supported text metrics, paired comparisons and blinded forms | Per-report scores, summaries, intervals, failure analysis, PUBLIC/PRIVATE review bundles |

Before coding, confirm the MIMIC-CXR candidate extraction file and its row linkage. Reuse Experiment 4's verified lab mapping and bank metadata where appropriate, but do not import its learned multimodal projection and describe it as raw HuBERT/RAD-DINO matching.

## 12. Reliability and acceptance checks

Implement per-query atomic checkpoints, writer locks, content-hashed protocols, successful-record skipping and explicit retries for failures. Use a configurable time budget below the remaining Biowulf allocation, plus STOP files and signal handling. Changed settings create new versions; no automatic budget escalation or silent exclusions.

Required tests before a full run:

- Deliberate patient/admission mismatch is rejected.
- Same-patient CXR and duplicate-study candidates are excluded.
- Reverse retrieval excludes forward-anchor patients in the primary score.
- Cached and exact reference searches agree on a small deterministic fixture.
- No cross-space vector comparison is permitted.
- Lab masks, column order and missingness survive packet construction.
- Evidence paths retain original sources and polarity; repeated paths do not multiply support.
- Graph selection cannot read targets; changing evaluation labels leaves retrieval and packets unchanged.
- Empty query concepts, no graph support and insufficient eligible candidates produce documented fallbacks.
- Fallback: retain available forward/bidirectional evidence with graph support marked absent; use fewer than k_CXR references if necessary, never invent them. Record every affected query.
- Interruption/resume produces the same completed outputs without duplicate successful cases.
- Reviewer exports omit model/arm labels and query reference targets.

## 13. Decision gates and immediate work

**Gate 1 — Input readiness:** verify MIMIC-CXR embeddings/candidates, training eligibility and patient crosswalk; establish aligned labs and compatible RAD-DINO features. Do not generate new reports until this passes.

**Gate 2 — Retrieval pilot:** inspect complete forward and reverse paths for 20 validation ECGs. Confirm that reverse scores are not trivial seed returns, that patient exclusions work and that the selected CXR evidence is interpretable.

**Gate 3 — Generation pilot:** run a small fixed validation sample across all arms for each model. Resolve formatting, context and completion problems without consulting test performance. Freeze the protocol afterward.

**Gate 4 — Internal comparison:** report D–C, C–B and completion rates, with parser and clinical-review limitations. Do not describe the already examined internal dataset as a new confirmatory test.

**Gate 5 — External evaluation:** apply the unchanged protocol to the prespecified Shaoxing sample. Freeze external label mapping before scoring and prohibit graph/encoder adaptation.

A graph benefit is supported only if D improves over C with uncertainty and missingness addressed, and the structure control does not explain the same gain. Improvement of C over B alone supports bidirectional retrieval, not graph inference. Improvement over A alone does not identify which component helped. If gains are absent, retain the simpler method and report the negative result.

**Immediate implementation priority:** BR00 and BR01. The current Shaoxing HuBERT encoding can continue independently; it supplies compatible query features for this extension.

## 14. Final CXR reference-count sensitivity

Implement the `k_CXR = 1,2,3,4,5` sweep using Experiment 02's BR06 notebook after BR02. Reuse the exact same anchors, initial CXR pools, reverse scores and graph scores. Derived child runs have separate signed configs and outputs, preserving the parent run. No encoding or reference extraction is repeated.

Keep the total context cap fixed. Allocate one common per-source findings cap across all five conditions, with room reserved for labs and graph associations; the default configuration gives 64 finding tokens per source. Record requested, selected and actually prompted reference counts. Compare all k values on a common completed-query cohort and report failure rates. A larger selected k must not be assumed to expose more findings when context is truncated.

Explore k on validation with one generator before expanding the workload. Freeze k before test/external scoring. Adjust the primary contrast family across generators and k when reporting the full sweep. Do not claim the maximum observed test score establishes the best k. See `experiment02/K_SENSITIVITY_README.md` for implementation details.
