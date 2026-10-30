# Foundation-model and generation integration

The notebooks operate on exports so that the clinical inference environment remains separate from lightweight graph development. They do not provide model-specific raw ECG/image preprocessing. Reuse the approved MedGemma and PULSE inference stacks and verify exact checkpoint/processor versions there.

## Candidate extraction

1. Establish patient splits and selected studies before any fitting.
2. Read only the selected sensor for that modality, with a fixed prompt requesting observations, uncertainty and negation.
3. Save the visible specialist output, exact model revision, prompt hash, processor revision and actual input checksum.
4. Parse output into a predefined clinical vocabulary and polarity. Preserve quoted source spans. Freeze the parser based on training/validation review. Do not expose target reports in the prompt or repair candidate outputs using test labels.
5. Create embeddings with a frozen modality encoder using identical preprocessing for references and queries. Document pretraining data and overlap uncertainty.
6. Export records and independent targets using DATA_CONTRACT.md.

## Report generation adapter

After notebook 05, consume `generation_packets.json`. For every packet:

- Resolve `query_id` to the same query sensor and preprocessing in all arms.
- Use the packet's candidate evidence and reference IDs; do not load target reports.
- Use an identical frozen generator and decoding settings for that modality. Keep a fixed tokenizer-based reference budget; the notebook's character cap is only preliminary packet preparation.
- Generate a report grounded in the query sensor. Retrieved examples concern other patients and are analogies, not new observations of this patient.
- Save query ID, arm, full report, actual reference IDs, model/processor revisions, prompt hash and inference settings.

The sensor-only arm also receives the same specialist candidates as the other arms, but no reference examples. This holds query evidence constant across comparisons.

Do not fabricate report text when the inference server is unavailable. No API keys or external endpoints are configured by these notebooks.

## Evaluation handoff

Use the private review key to attach generated reports and independently derived reference findings to randomized review IDs. Reviewers receive the completed review template without system/arm names; keep the private key separate. Empty templates are not completed evaluations.

`core.score_report_findings(reference, predicted)` compares independently annotated generated findings against known reference states. It returns TP/FP/FN, F1 on the evaluated concept subset, and unassessed predicted concepts. Compute per-concept counts and macro F1 separately if using macro F1 as a prespecified endpoint; the helper's single-record F1 is not macro F1. Add patient-paired uncertainty and clinically significant errors/omissions. Missing or uncertain reference findings cannot certify hallucination or correctness.
