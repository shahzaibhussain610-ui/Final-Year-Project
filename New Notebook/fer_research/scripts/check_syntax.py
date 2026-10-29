"""
Quick syntax check: extract all code cells from the notebook and
attempt to compile them to catch obvious Python syntax errors.
"""
import json, ast, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/facial_emotion_research.ipynb', encoding='utf-8'))
errors = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    try:
        ast.parse(src)
        print(f'Cell {i:2d}: OK')
    except SyntaxError as e:
        print(f'Cell {i:2d}: SYNTAX ERROR – {e}')
        errors += 1

print(f'\n{"All OK" if not errors else f"{errors} errors found"}')
