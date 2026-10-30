# Release validation

Local validation for this source release:

- 58 Python source files parsed successfully.
- 46 notebooks passed nbformat validation; 161 code cells compiled without executing real-data/model workflows.
- No executed notebook outputs, attached notebook media, data/weight formats, or common credential-token patterns were found by the release checker.
- 36 CPU fixture tests ran successfully; one optional FAISS equivalence test skipped because FAISS was not installed.
- Six clinical test modules were not loaded because the original `clinical_review` helper package is unavailable. They are not counted as passing tests.
- GPU checkpoint loading, original feature generation, real-data inference, and clinical review were not rerun for this packaging task.

Run `python tools/check_release.py` and `python tools/run_cpu_tests.py` to repeat available checks. CI uses those same checks. Local success does not imply a remote CI run has completed.
