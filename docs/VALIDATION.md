# Release validation

Local validation for this source release:

- 60 Python source files parsed successfully.
- 46 notebooks passed nbformat validation; 161 code cells compiled without executing real-data/model workflows.
- No executed notebook outputs, attached notebook media, data/weight formats, or common credential-token patterns were found by the release checker.
- 71 CPU tests were discovered: 56 passed and 15 were skipped.
- Skips: one optional FAISS equivalence test (FAISS not installed), plus 14 historical rating/adjudication notebook tests requiring private review images and completed forms that are deliberately not distributed.
- Synthetic clinical tests passed for blinding, independent-interpretation release gates, batch reference import, archive integrity, adjudication, preservation of initial judgments and unresolved-case denominators.
- GPU checkpoint loading, original feature generation, real-data inference, and clinical review were not rerun for this packaging task.

Run `python tools/check_release.py` and `python tools/run_cpu_tests.py` to repeat available checks. CI uses those same checks. Local success does not imply a remote CI run has completed.
