# Real export contract

Two JSON arrays are required. Use `core.fixture()` only to inspect the schema; its synthetic fields and labels cannot be used as research data.

## records.json — model inputs

One selected ECG and one selected CXR record per included admission:

```json
{
  "id": "admission123_ecg",
  "patient_id": "patient123",
  "admission_id": "admission123",
  "split": "train",
  "modality": "ecg",
  "embedding": [0.1, 0.2, 0.3],
  "embedding_space": "encoder-checkpoint-preprocessing-version",
  "synthetic": false,
  "candidates": [{
    "concept": "reviewed_ontology_concept",
    "polarity": "present",
    "model": "PULSE-ECG/PULSE-7B",
    "revision": "exact-checkpoint-revision",
    "sensor_sha256": "64-character-sha256-of-actual-sensor-input",
    "text": "Full specialist output",
    "quote": "specialist output"
  }]
}
```

For CXR use modality `cxr` and model `google/medgemma-1.5-4b-it`. The accepted identifiers describe this study configuration; update the protocol deliberately if using another checkpoint. Verify model access and preprocessing in the approved inference environment. Pin processor, prompt and ontology-parser revisions in an accompanying export manifest. Candidate quote inclusion is validated; correctness of the sensor checksum and interpretation must be audited against upstream files. This validator does not read raw waveforms/images.

Splits are `train`, `val`, `test`; polarity is `present`, `absent`, `uncertain`. Embedding dimensions/space must agree within a modality. Cross-modality spaces may differ. No test report, target finding, adjudication or target-derived embedding belongs in this file. Candidates must come from query sensors, not reference reports. Keep additional sensitive data in the approved source environment.

Native admission linkage, acquisition timestamps, study-selection rules, missing-case accounting, sensor paths, encoder/prompt/processor versions and sensor-hash verification belong in an accompanying provenance manifest. They are required scientific audit work, not inferred by this example adapter. Preserve failed/missing cases in the cohort flow diagram instead of silently dropping them.

## targets.json — independent reference annotations

```json
{
  "id": "admission123_ecg",
  "findings": {"reviewed_ontology_concept": 1, "another_concept": 0, "uncertain_concept": null},
  "target_source": "linked_independent_report_or_expert_annotation",
  "target_revision": "annotation-protocol-and-parser-version"
}
```

`1` means present, `0` explicitly absent, and `null` or missing means unknown. Do not convert every unmentioned finding to 0. Each record needs a target row; empty findings remain unevaluable rather than silently excluded. Do not use PULSE/MedGemma candidate outputs as their own targets. Report-based extraction is a noisy reference standard that needs human audit. Pairing itself is not a finding label.

Report generation outputs should preserve `query_id`, `arm`, `generated_report`, model revision, prompt/processor revision and actual reference IDs. Annotate generated and reference reports independently and blinded to arm; use `score_report_findings` for the explicitly known finding subset. It does not replace full-report clinical review.

## Notebook 01 assembly interface

When the two final exports are absent, notebook 01 accepts these four JSON arrays with exact one-to-one `id` coverage:

- `manifest`: `id`, `patient_id`, `admission_id`, `split`, `modality`.
- `embeddings`: `id`, `embedding`, `embedding_space`.
- `candidates`: `id`, `candidates`, where each candidate follows the schema above.
- `annotations`: `id`, `findings`, `target_source`, `target_revision`.

Set their paths in `SOURCE_PATHS` in notebook 01, or in a matching `export_sources` object in config.json. Existing final exports take precedence. The assembler preserves prespecified patient splits, copies only approved fields into model inputs, validates the full cohort, then creates both final files without overwriting existing exports. It does not infer targets from specialist candidates or generate embeddings.
