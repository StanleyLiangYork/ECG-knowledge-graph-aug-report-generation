"""Context-budget child runs: retain frozen evidence, change only common text caps."""
import copy
from pathlib import Path
from common import (assert_config, atomic, freeze_stage, immutable, parent_setup, read, sha, signature)
from packets import format_prompt

CAP_KEYS = ('context_tokens', 'finding_tokens_per_source', 'lab_tokens_per_source', 'relation_tokens')


def pulse_meter(runtime, image_path):
    """Same flat-patch/image-token expansion as the verified PULSE inference adapter."""
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    model = runtime['model']; tok = runtime['tokenizer']
    if getattr(model.config, 'mm_patch_merge_type', 'flat') != 'flat':
        raise ValueError('Unsupported patch merge: use the verified PULSE adapter')
    with Image.open(image_path) as im: image = im.convert('RGB')
    images = process_images([image], runtime['processor'], model.config)
    tensor = images[0] if isinstance(images, list) or images.ndim == 5 else images
    visual_tokens = int(tensor.shape[0]) * int(model.get_vision_tower().num_patches)
    marker = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
              if getattr(model.config, 'mm_use_im_start_end', False) else DEFAULT_IMAGE_TOKEN)
    def count(prompt):
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], marker + '\n' + prompt)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX)
        return len(ids) - 1 + visual_tokens
    return count, dict(image_size=list(image.size), visual_tokens=visual_tokens, context=int(runtime['context']),
                       revision=runtime['revision'], model='PULSE-ECG/PULSE-7B', output_tokens=None)


def _caps(cfg, cap):
    relation = min(cfg['relation_tokens'], max(48, (cap // 64) * 12))
    lab = min(cfg['lab_tokens_per_source'], max(24, cap // 16))
    finding = min(cfg['finding_tokens_per_source'], (cap - relation - cfg['k_anchors'] * lab - 64)
                  // (cfg['k_anchors'] + cfg['final_cxr']))
    if finding < 32: return None
    return dict(context_tokens=cap, finding_tokens_per_source=finding,
                lab_tokens_per_source=lab, relation_tokens=relation)


def prepare_context_fit(cfg, runtime, version='v1', max_new_tokens=384, frozen_context_config=None):
    """Inspect every case's text budget; preserve all candidate pools and evidence.

    Calibrate in validation. For later cohorts pass the validation child config to
    freeze the selected caps rather than selecting them again. Native inference
    still checks the actual processed image budget for every query.
    """
    from queries import load_queries, image_provider
    from packets import prepare_packets
    assert_config(cfg); parent_setup(cfg)
    if runtime.get('family') != 'pulse': raise ValueError('Calibrate with the working PULSE runtime first')
    if not version or Path(version).name != version: raise ValueError('Simple version name required')
    if type(max_new_tokens) is not int or max_new_tokens < 1: raise ValueError('Positive output budget required')
    source = Path(cfg['run_dir']).resolve(); cohort = cfg['cohort']
    queries = load_queries(cfg); manifest = read(source / 'packets' / cohort / 'manifest.json')
    if manifest['query_ids'] != [q['id'] for q in queries]: raise ValueError('Complete BR03 first')
    source_packets = {}
    for qid in manifest['query_ids']:
        path = source / 'packets' / cohort / (qid + '.json')
        if sha(path) != manifest['packets'][qid]: raise ValueError('Source packet changed')
        source_packets[qid] = read(path)
    image, _ = image_provider(cfg)(queries[0])
    count, meter = pulse_meter(runtime, image)
    meter['output_tokens'] = max_new_tokens
    if frozen_context_config:
        frozen = read(frozen_context_config)
        choices = [{key: frozen[key] for key in CAP_KEYS}]
    else:
        if cohort != 'internal_val': raise ValueError('Choose caps on internal_val; supply its frozen child config for later cohorts')
        choices = [c for n in sorted({cfg['context_tokens'], *range(256, cfg['context_tokens'] + 1, 64)}, reverse=True)
                   if (c := _caps(cfg, n)) is not None]
    selected = None; counts = []; safety = 32
    for caps in choices:
        trial = dict(cfg, **caps); counts = []; valid = True
        for qid, packet in source_packets.items():
            for case in packet['cases']:
                prompt, exposure = format_prompt(case, runtime['tokenizer'], trial)
                expanded = count(prompt)
                if case['arm'] in ('bidirectional_graph', 'bidirectional_shuffled_graph'):
                    # A smaller cap must not silently remove the graph explanation slot.
                    valid &= any(s['slot'] == 'relations' and s['included'] for s in exposure['slots'])
                valid &= expanded + max_new_tokens + safety <= meter['context']
                counts.append(dict(query_id=qid, arm=case['arm'], expanded_input_tokens=expanded,
                    context_tokens=exposure['context_tokens'], remaining_tokens=meter['context'] - expanded - max_new_tokens))
        if valid: selected = caps; break
    if selected is None:
        raise ValueError('No tested common caps fit every case with the graph slot retained. Do not increase the model context limit; inspect image-token cost.')
    child = source / 'context_fit' / version / cohort
    child_cfg = dict(cfg, **selected, run_dir=str(child))
    protocol = copy.deepcopy(read(source / 'protocol.json'))
    protocol['config'] = {k: v for k, v in child_cfg.items() if k != 'cohort'}
    identity = dict(source_run=str(source), parent_protocol_sha256=sha(source / 'protocol.json'),
        source_packet_manifest_sha256=sha(source / 'packets' / cohort / 'manifest.json'),
        caps=selected, meter=meter, safety_tokens=safety, helper_sha256=sha(__file__),
        frozen_context_config_sha256=sha(frozen_context_config) if frozen_context_config else None,
        policy='Same frozen pools, rankings, selected IDs and complete evidence cases; common prompt caps for all generators. Calibration image size is checked again before each query.')
    protocol['context_fit'] = identity
    immutable(child / 'protocol.json', protocol)
    link = child / 'banks'
    if not link.exists(): link.symlink_to(source / 'banks', target_is_directory=True)
    if link.resolve() != (source / 'banks').resolve(): raise ValueError('Unexpected shared bank path')
    immutable(child / 'queries' / cohort / 'manifest.json', read(source / 'queries' / cohort / 'manifest.json'))
    if cohort == 'shaoxing':
        import shutil
        original = source / 'queries' / cohort / 'candidates.jsonl'; target = child / 'queries' / cohort / 'candidates.jsonl'
        if target.exists():
            if sha(target) != sha(original): raise ValueError('Query candidates changed')
        else: shutil.copyfile(original, target)
    for name in ('queries_' + cohort, 'query_extraction_' + cohort, 'indexes', 'banks_ready'):
        path = source / 'stages' / (name + '.json')
        if path.exists(): freeze_stage(child, name, read(path)['inputs'])
    original_stage = read(source / 'stages' / ('retrieval_' + cohort + '.json'))
    sign = freeze_stage(child, 'retrieval_' + cohort, original_stage['inputs'])
    for qid in manifest['query_ids']:
        path = source / 'retrieval' / cohort / (qid + '.json'); r = read(path)
        if r['protocol_signature'] != signature(original_stage): raise ValueError('Source retrieval stage mismatch')
        r['protocol_signature'] = sign
        immutable(child / 'retrieval' / cohort / (qid + '.json'), r)
    prepare_packets(child_cfg)
    atomic(child / 'config.json', child_cfg)
    atomic(child / 'audits/context_fit.json', dict(identity=identity, queries=len(queries), cases=counts))
    return dict(config=child_cfg, config_path=str(child / 'config.json'), caps=selected,
        maximum_input_tokens=max(r['expanded_input_tokens'] for r in counts),
        visual_tokens=meter['visual_tokens'], model_context=meter['context'])


def generate_context_fit(cfg, runtime, **settings):
    """Keep ordinary leakage gate; fail before inference if image dimensions differ."""
    import queries
    from unittest.mock import patch
    from extensions.leakage_controls import guarded_generate
    meta = read(Path(cfg['run_dir']) / 'protocol.json')['context_fit']
    if sha(__file__) != meta['helper_sha256']: raise ValueError('Context-fit helper changed')
    if runtime.get('family') == 'pulse':
        if runtime['revision'] != meta['meter']['revision'] or int(runtime['context']) != meta['meter']['context']:
            raise ValueError('PULSE context/checkpoint differs from calibration')
        if settings.get('max_new_tokens', 384) != meta['meter']['output_tokens']:
            raise ValueError('Use the calibrated PULSE output budget')
    original = queries.image_provider
    def provider_factory(config):
        provider = original(config)
        def provide(row):
            from PIL import Image
            image, tensor = provider(row)
            with Image.open(image) as im:
                if list(im.size) != meta['meter']['image_size']:
                    raise ValueError('Query image size differs from PULSE context calibration; stop before inference')
            return image, tensor
        return provide
    with patch.object(queries, 'image_provider', side_effect=provider_factory):
        return guarded_generate(cfg, runtime, **settings)
