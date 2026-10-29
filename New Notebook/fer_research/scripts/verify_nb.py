import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('notebooks/facial_emotion_research.ipynb', encoding='utf-8'))
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    ct = c['cell_type']
    src = c['source']
    first_line = (src[0] if src else 'empty')[:70]
    print(f'Cell {i:2d} ({ct}): {first_line!r}')
