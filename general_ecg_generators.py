"""Additive general-purpose ECG generators; existing signed helpers stay unchanged."""
from pathlib import Path
from report_study import read, sha, signature
import ecg_external_generation as engine

MODELS = {
    'llama32_base': ('meta-llama/Llama-3.2-11B-Vision-Instruct', 'llama32_11b_base_v1', 'llama'),
    'qwen38_27b': ('Qwen/Qwen3.8-27B', 'qwen38_27b_nonthinking_v1', 'qwen'),
}


def prepare(base, key):
    base = Path(base)
    model_id, folder, backend = MODELS[key]
    plan = read(base / 'study_protocol.json')
    all_cases = read(base / 'generation_cases.json')
    if signature(all_cases) != plan['cases_sha256']:
        raise ValueError('Base case hash mismatch')
    cases = [c for c in all_cases if c['modality'] == 'ecg']
    if not cases:
        raise ValueError('No ECG cases')
    plan = dict(plan, cases_sha256=signature(cases), parent_cases_sha256=signature(all_cases),
                general_generator={'key': key, 'model_id': model_id,
                                   'helper_sha256': sha(__file__), 'enable_thinking': False})
    study = base / folder
    engine.immutable(study / 'study_protocol.json', plan)
    engine.immutable(study / 'generation_cases.json', cases)
    return study


def validate_qwen_config(config):
    text = config.get('text_config', {})
    if (config.get('model_type') != 'qwen3_5' or not config.get('vision_config')
            or text.get('hidden_size') != 5120 or text.get('num_hidden_layers') != 64):
        raise ValueError('Expected Qwen3.8-27B multimodal architecture; verify MODEL_PATH')


def load(key, path, load_in_4bit=False):
    model_id, _, backend = MODELS[key]
    path = Path(path).resolve()
    if backend == 'llama':
        runtime = engine.load_llama(path, load_in_4bit=load_in_4bit)
        runtime['model_id'] = model_id
        return runtime
    import torch
    import transformers
    validate_qwen_config(read(path / 'config.json'))
    if not torch.cuda.is_available():
        raise RuntimeError('Use a CUDA GPU allocation')
    cls = getattr(transformers, 'AutoModelForMultimodalLM', None)
    if cls is None:
        cls = getattr(transformers, 'AutoModelForImageTextToText', None)
    if cls is None:
        raise ImportError('Install a Transformers version supporting Qwen3.8 in its own virtual environment')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = transformers.AutoProcessor.from_pretrained(str(path), local_files_only=True)
    kwargs = dict(local_files_only=True, device_map='auto', torch_dtype=dtype)
    if load_in_4bit:
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = cls.from_pretrained(str(path), **kwargs).eval()
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Unloaded model parameters')
    context = getattr(getattr(model.config, 'text_config', model.config), 'max_position_embeddings', None)
    if not isinstance(context, int):
        raise ValueError('Cannot determine context limit')
    revision = path.name
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        revision = signature({str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()})
    return dict(model=model, processor=processor, tokenizer=processor.tokenizer,
                revision=revision, dtype=dtype, context=context, modality='ecg',
                model_id=model_id, family=backend, enable_thinking=False, load_in_4bit=load_in_4bit)


def generate(study, runtime, provider, **kwargs):
    study = Path(study)
    plan = read(study / 'study_protocol.json')
    settings = plan['general_generator']
    if settings['helper_sha256'] != sha(__file__):
        raise ValueError('Helper changed; use a new versioned output folder')
    model_id, _, backend = MODELS[settings['key']]
    if (runtime['model_id'] != model_id or runtime['family'] != backend
            or runtime.get('enable_thinking', False)):
        raise ValueError('Runtime does not match the nonthinking study')
    if backend == 'llama':
        tuned = read(study.parent / 'ecg_instruct_llama32_11b_v2' / 'ecg_generation_protocol.json')
        if tuned.get('load_in_4bit', False) != runtime.get('load_in_4bit', False):
            raise ValueError('Match LOAD_IN_4BIT to the completed ECG-Instruct run')
        actual = engine.configure_decoding(runtime['model'].generation_config, plan, 'llama').to_dict()
        for k in ('do_sample', 'num_beams', 'repetition_penalty', 'no_repeat_ngram_size'):
            if actual.get(k) != tuned['decoding'].get(k):
                raise ValueError('Base/tuned Llama decoding differs: ' + k)
        for k in ('max_new_tokens', 'reference_token_cap'):
            if plan[k] != tuned['study'][k]:
                raise ValueError('Base/tuned Llama budgets differ: ' + k)
    return engine.generate(study, runtime, provider, **kwargs)
