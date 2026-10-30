RAD-DINO borderline equivalence recheck

Biowulf audit: 9/10 probes passed. Symile admission 26559734 had
relative L2=0.0049690749 (passes <=0.005) and cosine=0.9999878407
(just below >=0.99999). All MIMIC probes passed. This suggests a small
numerical discrepancy; its cause is not established by the audit alone.

Upload ONLY BR01A_batch_matched_rad_recheck.ipynb into experiment02.
Run it in the original RAD-DINO CUDA environment instead of rerunning the
single-image BR01A. It keeps the original thresholds and reproduces the
Symile encoder's original unique-admission order, first single probe and
subsequent batches (default 16). If the original job overrode RADDINO_BATCH_SIZE,
set ORIGINAL_BATCH_SIZE to that value. MIMIC verification remains unchanged.

No frozen Python helper is replaced. Prior failed audits are preserved in
audits/rad_history/. The current audit records batch details and this notebook
hash. No new run_dir or repeat bank preparation is required for this recheck.
Only proceed to BR01 finalization when every probe passes. If it still fails,
inspect the new audit and original environment/processor/attention backend;
do not lower thresholds or set passed=True manually.

Local tests verify notebook validity and original unique-admission batch mapping
with unsorted/duplicate admissions using mocked inference. GPU success remains
to be verified on Biowulf.
