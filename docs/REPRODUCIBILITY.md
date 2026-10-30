# Release scope and reproducibility status

This release packages the supplied local source tree. It is not a complete archival reproduction of every Biowulf runtime or a claim that every notebook has been executed on a clean machine.

## Known missing components

The original `experiment02/clinical_review/workflow.py` and `archived_inputs.py` are now included unchanged from the supplied source files. They implement the clinical-review workflow and verified historical-image adapter. Completed participant/reviewer records remain excluded.

Original upstream feature-bank production code, protected data, checkpoints, immutable run artifacts, full historical environment lockfiles and manuscript analysis-output tables are not included. Root 00c consumes pre-existing feature arrays. Real reproduction requires those independently obtained inputs and matching provenance.

## Public-copy changes

- Cleared every notebook execution count/output, rich-output attachment and local notebook metadata.
- Replaced author-specific filesystem prefixes with explicit `/path/to/...` placeholders. Original study files were not modified.
- Clinical-review notebooks identify the included helpers and exclusion of completed reviewer data.
- Updated BR04B's public entry point to invoke the existing verified image-archive wrapper for Shaoxing, matching the external workflow described in the study.
- Set the public BR04B and BR05A external base-Llama routing to the 1,024-token analysis run. Internal base-Llama remains 384.
- Added README, MIT license, environment/data notes, validation tools and source hashes.
- Historical rating/adjudication tests explicitly skip when their private fixtures are absent. Clinical helpers are unchanged; no completed review data were added.

Source/output hashes are recorded in `source_manifest.json`. Public sanitization changes some hashes. Never use this copy to resume or bypass a historical run's immutable code/protocol checks. Keep original signed source and run artifacts separately.

## Validation scope

`tools/check_release.py` validates notebook structure, compiles code cells/Python source without model execution, checks for notebook outputs and common credential patterns, and enforces exclusion of patient-data formats. `tools/run_cpu_tests.py` runs Experiment 02 tests, including clinical workflow and import tests, with artificial fixtures and mocked model inference. Fourteen historical rating/adjudication notebook tests skip without private review fixtures; these are not counted as passes. Optional FAISS checks can skip if FAISS is not installed.

Neither passing tests nor provenance hashes establish clinical validity, causal inference, complete data independence, model-pretraining independence, or resolution of parser/rendering limitations. All clinical review in the manuscript is calibration/exploratory unless a separately documented main review was conducted.
