"""Validate the public source tree without executing data/model notebooks."""
from pathlib import Path
import ast
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.ipynb_checkpoints', '.venv-cpu', 'validation_reports'}
SECRET = re.compile(r'(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
errors = []
counts = {'python_files': 0, 'notebooks': 0, 'code_cells': 0}
for path in ROOT.rglob('*'):
    if not path.is_file() or any(part in EXCLUDED or part.startswith('.venv') for part in path.relative_to(ROOT).parts):
        continue
    relative = str(path.relative_to(ROOT))
    if path.suffix in {'.jsonl', '.npy', '.npz', '.mat', '.hea', '.dat', '.dcm', '.pt', '.pth', '.safetensors'}:
        errors.append(f'Unexpected data/weight artifact: {relative}')
    if path.suffix not in {'.py', '.ipynb', '.md', '.txt', '.json', '.cff', '.yml'}:
        continue
    text = path.read_text()
    if SECRET.search(text):
        errors.append(f'Possible credential: {relative}')
    if path.suffix == '.py':
        ast.parse(text, filename=relative)
        counts['python_files'] += 1
    elif path.suffix == '.ipynb':
        notebook = nbformat.reads(text, as_version=4)
        nbformat.validate(notebook)
        counts['notebooks'] += 1
        for i, cell in enumerate(notebook.cells):
            if cell.get('attachments'):
                errors.append(f'Unexpected attachment: {relative}:{i}')
            if cell.cell_type == 'code':
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'Executed cell: {relative}:{i}')
                # Convert notebook magics before syntax checking, without executing them.
                code = cell.source
                if any(line.lstrip().startswith(('%', '!')) for line in code.splitlines()):
                    from IPython.core.inputtransformer2 import TransformerManager
                    code = TransformerManager().transform_cell(code)
                compile(code, f'{relative}:cell-{i}', 'exec')
                counts['code_cells'] += 1
print(json.dumps({'checks': counts, 'errors': errors}, indent=2))
raise SystemExit(bool(errors))
