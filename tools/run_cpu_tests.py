"""Run available Experiment 02 fixture tests and disclose unavailable clinical tests."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiment02'
sys.path[:0] = [str(EXP), str(ROOT)]
modules = []
unavailable = []
for path in sorted(EXP.glob('test_*.py')):
    if ('clinical_review' in path.read_text()) and not (EXP / 'clinical_review/workflow.py').is_file():
        unavailable.append(path.stem)
    else:
        modules.append(path.stem)
print('Unavailable original clinical-helper tests:', ', '.join(unavailable) or 'none', flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
