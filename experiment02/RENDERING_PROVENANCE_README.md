# BR05 rendering provenance: identical waveforms, different PNG files

The observed internal-validation audit has one waveform tensor hash per query but two image-file hashes: PULSE uses NumPy1.26.4/Pillow10.4.0/Matplotlib3.9.4, while Llama-ECG, base Llama and Qwen use NumPy2.4.6/Pillow12.3.0/Matplotlib3.11.1. This excludes a recorded tensor mismatch; it does not establish identical historical pixels. PNG metadata, compression, fonts, layout and antialiasing can affect file identities. Do not assume a metadata-only cause without evidence.

Run **BR05A_rendering_provenance_and_within_generator_evaluation.ipynb** using the same context-fit child config and report folders as BR05. It does not edit frozen root Python files or existing reports.

## Immediate route: native within-generator evaluation

Run section A in the CPU evaluation environment for `pulse`, `llama_ecg`, `llama_base`, and `qwen38` in turn. The original evaluator checks identical waveform and PNG hashes across the selected generator's arms. Each result retains its own complete-case denominator, pairwise comparisons, failure rates and completion-sensitive F1. These are valid comparisons of graph versus control under that generator's fixed rendering; they do not establish identical images across models.

Outputs:

```text
<child_run>/evaluation/internal_val/within_generator_v1/<model>/
```

With the current coverage, all-arm-complete counts should be750 for PULSE, Llama-ECG and Qwen, and672 for base Llama. Verify the actual output. Do not rank models as though these were one common-case, identical-image comparison. Differences in rendering and completion can affect cross-model conclusions.

## Optional route: establish whether PNG encodings alone differ

1. Open BR05A in the original PULSE environment. Run setup/folder definitions, skip scoring if dependencies are unavailable, then section B with `RENDER_GROUP='pulse'` and `MAX_NEW_RENDERS=3`. Inspect the pilot and use0 for the remaining queries.
2. Open the same notebook in the original newer environment. Repeat section B with `RENDER_GROUP='modern'`. No GPU model needs loading. Do not upgrade libraries to make versions agree: reproduce each historical environment instead.
3. The helper validates tensor/admission identity and reproduces each historical PNG byte hash, including the source identities recorded for failed reports. If a historical hash cannot be reproduced, stop and inspect `first_reconstruction_failure.json`. Matching library versions alone cannot certify rendering history; fonts and rc settings can matter.
4. Section C compares exact decoded RGBA identities for every reconstructed ECG. The images are retained in separate directories, not the shared generation cache. Resumption checks existing files before skipping completed queries.
5. If ALL pixels match, section D permits a combined analysis with canonical pixel identities in memory. Native protocol, response and prompt validation still run. A separate immutable equivalence protocol records the reconstruction evidence. No source journal or report hash is edited. Output goes to a new `v2_exact_pixel_equivalence` evaluation version.
6. If ANY pixels differ, section D stops. Keep within-generator results and disclose the rendering difference. A definitive future cross-model comparison requires pre-rendering one immutable query-image set and recording those same images for every generator before inference. A retrospective image replacement cannot change what earlier models saw.

Historical reconstruction outputs:

```text
<child_run>/audits/historical_rendering/internal_val/v1/pulse/
<child_run>/audits/historical_rendering/internal_val/v1/modern/
<child_run>/audits/historical_rendering/internal_val/v1/pixel_comparison.json
```

Original rendering libraries are required, but model weights/Transformers loading are not. The reconstruction helper currently supports internal Symile cohorts only. A separate adapter is needed for Shaoxing's waveform/image provenance. Each saved image/check is resumable; a STOP file and a two-hour run budget stop between cases. Keep only one writer per rendering-group directory. Budget extra disk space for retained reconstructed images.

## Clinical review implications

The existing clinical helper also checks exact image-file identities. Do not disable that check or give the clinician a newly rendered image while claiming it was the generator's original input. Retain reconstructed historical images for any subsequent review extension. If pixels differ across groups, clinical packages need generator-specific historical images and explicit source access; the current mixed-model BR07 package cannot silently combine them. Even if decoded pixels match, mixed-model review requires an explicit pixel-equivalence-aware image provider rather than deleting the hash check.

CPU tests: `python -m unittest test_rendering_audit_extension -v`. Fixtures test metadata-only PNG differences, true pixel differences, historical-hash failure, resumption, changed journals and exact-equivalence evaluation without rewriting originals. They do not establish equivalence for the real Biowulf inputs; that requires the two-environment audit above.
