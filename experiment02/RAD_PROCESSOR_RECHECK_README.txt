RAD-DINO processor backend recheck

The original and batch-matched audits returned the same values. This rules out
batch size as the observed explanation in that recheck. Current Biowulf environment
returns BitImageProcessor by default and BitImageProcessorPil for backend='pil'.
Transformers documentation describes automatic torchvision selection when available:
https://github.com/huggingface/transformers/blob/main/docs/source/en/image_processors.md

Upload BR01A_original_processor_rad_recheck.ipynb ONLY; no frozen Python source
changes are required. Restart the RAD-DINO kernel and run all cells of this
notebook instead of the earlier BR01A/recheck notebooks. Keep original batch size
16 unless your original encoding job used a documented override. Keep attention
implementation None. The notebook explicitly chooses backend='pil', use_fast=False.

It first repeats the original five probes, then expands to 20 per source. Both
sets must pass the ORIGINAL relative-L2 <=0.005 and cosine >=0.99999 limits.
It preserves old audits in rad_history, records actual processor/model/runtime
settings and saves its notebook hash. It does not change weights or feature arrays.
If this still fails, investigate original environment/attention/preprocessing
versions; do not override passed or weaken thresholds. GPU verification is pending.

CPU tests validate original unique-admission batch indexing, explicit processor
selection and notebook syntax. The original backend's historical version is not
recorded in feature metadata; numerical reproduction must establish compatibility.
