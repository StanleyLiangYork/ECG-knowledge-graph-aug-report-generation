# Final CXR reference-count sensitivity: k = 1, 2, 3, 4, 5

Use **BR06_cxr_reference_k_sensitivity.ipynb** after BR02 has completed. Continue the current BR01B and bank-preparation work first. The new notebook does not replace any existing BR00–BR05 notebook or root Python helper.

Upload BR06 and the `extensions/k_sweep.py` file into the same Experiment 02 folder. The subfolder is intentional: the original source-freeze checks remain valid. No changes to the current config.json or original run directory are required.

## Meaning of k

This sweep changes **final_cxr**, the number of distinct-patient MIMIC-CXR references selected for the report context. The ECG/Symile admission anchor count stays fixed (default three). Forward per-anchor retrieval remains 20, reverse admission count remains three, and all original candidate scores are reused. Fewer references are used if the pool has insufficient distinct patients; every such query is audited.

The initial candidate pool, ECG queries, training bank, graph, extractors and forward/reverse scores are identical across the sweep. No encoder calls, bank reconstruction or reference extraction are repeated.

## Context control

The total context cap stays fixed (default 1,024 tokens). To leave room for graph associations even at k=5, the sweep uses one common per-source findings cap across ALL k values. With the default three anchors and five maximum CXR references, this is 64 tokens per source instead of the parent's 96. Lab and relationship caps remain fixed. Thus k is the only varying setting **within the sweep**, but these outputs should not be substituted into the older k=2 run.

The formatter logs omissions. The summary reports requested k, selected references and actually included reference-source headers, plus mean context tokens. Higher k must not be assumed to mean every source observation survives truncation. Report this exposure alongside quality scores. No content is invented to pad shorter prompts.

## Running order

1. In BR06, run setup and preparation with `K_VALUES = [1, 2, 3, 4, 5]`.
2. Use one generator initially, preferably the prespecified development generator, and load it once.
3. Run `MAX_NEW_PER_K = 5` to pilot all arms at each k (25 reports in total).
4. Inspect all five conditions, then set `MAX_NEW_PER_K = 0` to complete them.
5. Use the graph/evaluation environment for evaluation and aggregation. Set the same completed model set at every k. Re-run only the setup/preparation cells to reconstruct configurations; model loading is unnecessary for evaluation.

Child runs are created under:

```text
<original_run_dir>/k_sensitivity/v1/<cohort>/k_1/
<original_run_dir>/k_sensitivity/v1/<cohort>/k_2/
...
<original_run_dir>/k_sensitivity/v1/<cohort>/k_5/
```

Each has its own signed config, retrieval selections, packets, generation journals and evaluation outputs. Banks are shared by a directory symlink; do not run BR00/BR01 writes inside a child. Query manifests/candidate journals are copied with provenance checks. The original retrieval and report run is preserved. Choose a new SWEEP_VERSION for changed sweep settings; do not edit old protocols.

All k values share one generation time budget measured from the sweep call. STOP/signal/CUDA failure ends the sweep between reports. The next allocation can resume existing successes. Changing a generation budget requires a new model/mode/budget run as in the main workflow.

## Interpretation and cost

Use internal validation to investigate k. Freeze the choice before evaluating internal test or Shaoxing performance; do not choose k from external F1 or reference diagnoses. A full sensitivity study on external/test data must report all k values as exploratory, rather than selecting the best test result.

Start with one generator. Four generators × five arms × five k values × 1,000 queries would produce 100,000 reports. The implementation does not copy old sensor-only reports across protocol signatures; each child is self-contained.

The aggregated outputs include:

- `summary_by_k.csv`: each k's existing complete-case summary (cohorts may differ).
- `common_query_summary_by_k.csv`: quality scores on the same jointly complete query set across all k/models/arms.
- `common_query_primary_effects.csv`: patient-paired graph-minus-bidirectional F1 effects on that shared set.
- `primary_effects_by_k.csv`: original per-k effects, labeled separately.
- `context_exposure_by_k.csv`: successful-report counts, actual context tokens, selected and prompted CXR references.
- `k_sensitivity.json`: shared query IDs and all summaries; no automatic best-k selection.

Primary-effect multiplicity adjustment is repeated across both generators and k values. The inherited sign-flip p-values remain exploratory and require their stated symmetry assumption. Generation failures and the reduced common cohort must be reported. ROUGE remains unavailable without independent reference text.

The existing optional retrieval-diagnostic helper labels metrics at k=2. Do not apply it to child runs and interpret those columns as MRR@k/nDCG@k. BR06 evaluates report quality; independent retrieval judgments would need a separately k-aware relevance analysis.
