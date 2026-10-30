# Release scope and reproducibility status

This release packages the supplied local source tree. It is not a complete archival reproduction of every Biowulf runtime or a claim that every notebook has been executed on a clean machine.

## Known missing components

The local `experiment02/clinical_review/` package was absent, including `workflow.py` and `archived_inputs.py`. The supplied `human_review` folder contained an import helper and completed assessment artifacts, not that package. The helper and BR07–BR09 notebook sources are included; clinical package-dependent notebooks/tests remain blocked until the original modules are restored. No replacement clinical algorithm has been invented.

Original upstream feature-bank production code, protected data, checkpoints, immutable run artifacts, full historical environment lockfiles and manuscript analysis-output tables are not included. Root 00c consumes pre-existing feature arrays. Real reproduction requires those independently obtained inputs and matching provenance.

## Public-copy changes

- Cleared every notebook execution count/output, rich-output attachment and local notebook metadata.
- Replaced author-specific filesystem prefixes with explicit `/path/to/...` placeholders. Original study files were not modified.
- Added dependency notices to clinical-review notebooks.
- Updated BR04B's public entry point to invoke the existing verified image-archive wrapper for Shaoxing, matching the external workflow described in the study.
- Set the public BR04B and BR05A external base-Llama routing to the 1,024-token analysis run. Internal base-Llama remains 384.
- Added README, MIT license, environment/data notes, validation tools and source hashes.

Source/output hashes are recorded in `source_manifest.json`. Public sanitization changes some hashes. Never use this copy to resume or bypass a historical run's immutable code/protocol checks. Keep original signed source and run artifacts separately.

## Validation scope

`tools/check_release.py` validates notebook structure, compiles code cells/Python source without model execution, checks for notebook outputs and common credential patterns, and enforces exclusion of patient-data formats. `tools/run_cpu_tests.py` runs the available non-clinical Experiment 02 tests with artificial fixtures and mocked model inference. Clinical-helper-dependent test modules are explicitly listed as unavailable. Optional FAISS checks can skip if FAISS is not installed.

Neither passing tests nor provenance hashes establish clinical validity, causal inference, complete data independence, model-pretraining independence, or resolution of parser/rendering limitations. All clinical review in the manuscript is calibration/exploratory unless a separately documented main review was conducted.
