# Recover the two failed BR01B reference images

The original extraction has 28,072 successful images and two generation failures. The token-limit error does not establish whether the text is simply long or repetitive; the other image explicitly failed repetition QC. Repeating identical deterministic settings need not fix either failure.

Upload the contents of `experiment02_cxr_recovery_update.zip` into:

```text
/path/to/repository/experiment02
```

Run `BR01C_recover_failed_cxr_candidates.ipynb` in the same MedGemma virtual environment/checkpoint used for BR01B. The first cells find the existing config and load that checkpoint. It requires every image to have an original attempt, so no untouched record can be silently lost.

The retry settings are 2048 output tokens, repetition penalty 1.15 and no-repeat 8-gram. The image-only prompt remains unchanged. Both original quality requirements remain enforced: complete generation and repetition QC. The notebook retries only original failures, retains every original successful extraction, and never modifies the original journal or frozen BR01B protocol. Do not rerun BR01B while using this recovery route: its journal hash is now an input to the recovery protocol.

Inspect the two reports after the retry cell. Require `combined_ok == total == 28074`, then run BR01C's final cell. This finalizer validates the original successes plus the separately signed recovery outputs and creates the standard banks/indexes. It **replaces BR01's original final cell** for this route. The original finalizer will still reject the failures recorded in the original journal; the new route does not hide or overwrite them.

Then continue to BR02, BR03, BR03A, BR04A and BR05. For Shaoxing, the usual BR02A query-candidate step is still needed before retrieval.

Default recovery outputs:

```text
/path/to/repository/outputs/real_run_001/experiment02_v1/extraction/recovery_t2048_r115_ng8_v1/protocol.json
/path/to/repository/outputs/real_run_001/experiment02_v1/extraction/recovery_t2048_r115_ng8_v1/cxr_candidates.jsonl
/path/to/repository/outputs/real_run_001/experiment02_v1/extraction/recovery_t2048_r115_ng8_v1/coverage.json
```

`stages/banks_ready.json` records the retry journal/protocol/helper hashes and recovered IDs. Recovered bank rows carry their recovery protocol identity. The updated `extensions/leakage_controls.py` checks these additional hashes during BR03A/BR04A. Preserve the `extensions` directory structure when copying the update. No original signed root Python file is changed.

Resume with the same notebook/settings after interruption. Successful retries are skipped, and a partially written final journal line is repaired. STOP files/signals/time budgets stop between images. If a retry remains unsuccessful, finalization stops; preserve its error instead of accepting truncated text, disabling QC or excluding the image. Changed retry settings require a separately named recovery version. These settings have not been tested on the two actual images locally.

For the manuscript, disclose that 2/28,074 reference images required generation-QC-triggered retries with different decoding. The retry policy uses generation validity, not clinical target labels. A successful generation/QC check does not establish clinical correctness.
