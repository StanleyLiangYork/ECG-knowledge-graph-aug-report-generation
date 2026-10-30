"""Inspect one failed generation without accepting it as a candidate report."""
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import sensor_candidate_generation as generation


def diagnose_cxr_generation(tasks, arrays, runtime, recovered_directory):
    if runtime['modality'] != 'cxr':
        raise ValueError('Run in the loaded MedGemma kernel')
    folder = Path(recovered_directory)
    latest = {r['id']: r for r in generation.read_lines(folder / 'cxr_sensor_only.jsonl')}
    failed_ids = {rid for rid, row in latest.items() if row['status'] != 'ok'}
    task = next((t for t in tasks if t['id'] in failed_ids), None)
    if task is None:
        raise ValueError('No failed CXR case found')
    diagnostics = folder / 'diagnostics'
    image_path = diagnostics / 'first_failed_cxr.png'
    generation.materialize(arrays[task['split']][task['source_index']], 'cxr', image_path)
    model, tokenizer = runtime['model'], runtime['tokenizer']
    original_generate = model.generate
    captured = {}

    def capture(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        captured['tokens'] = output[0, kwargs['input_ids'].shape[-1]:].detach().cpu().tolist()
        return output

    error = None
    # Observe the existing inference function without modifying source files,
    # generation settings, saved protocols, or candidate reports.
    with patch.object(model, 'generate', side_effect=capture):
        try:
            generation.infer(runtime, image_path, max_new_tokens=1024)
        except Exception as exc:
            error = str(exc)
    ids = captured.get('tokens', [])
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    grams = Counter(tuple(ids[i:i + 8]) for i in range(max(0, len(ids) - 7)))
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    markers = {}
    for token in ('<eos>', '<end_of_turn>'):
        token_id = tokenizer.get_vocab().get(token)
        markers[token] = {'id': token_id, 'occurrences': ids.count(token_id) if token_id is not None else 0}
    report = dict(id=task['id'], error=error, generated_tokens=len(ids),
                  generation_eos_ids=eos_ids, tokenizer_eos_id=tokenizer.eos_token_id,
                  eos_positions=[i for i, value in enumerate(ids) if value in eos_ids],
                  stop_markers=markers, last_token_ids=ids[-20:],
                  most_repeated_8_token_sequence_count=max(grams.values(), default=0),
                  generation_config=model.generation_config.to_dict(),
                  raw_output=raw)
    destination = diagnostics / 'first_failed_cxr.json'
    destination.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ('raw_output', 'generation_config', 'id')}, indent=2))
    print('\nBEGINNING OF OUTPUT:\n', raw[:1600])
    print('\nEND OF OUTPUT:\n', raw[-1600:])
    print('\nFull diagnostic saved:', destination)
    return report
