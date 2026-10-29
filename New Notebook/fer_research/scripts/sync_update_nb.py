import sys, os, json
sys.path.insert(0, os.path.abspath("."))
from scripts.apply_infra_fix import NEW_INFRA_SOURCE

with open("scripts/update_notebook.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if line.strip().startswith('CELL_INFRA_CODE = {'):
        start_idx = i
    elif start_idx is not None and line.startswith('}'):
        end_idx = i + 1
        break

if start_idx is not None and end_idx is not None:
    formatted_source = json.dumps(NEW_INFRA_SOURCE, indent=8)
    replacement = f"""CELL_INFRA_CODE = {{
    "cell_type": "code",
    "execution_count": None,
    "id": "exp_infra_code",
    "metadata": {{}},
    "outputs": [],
    "source": {formatted_source}
}}
"""
    new_lines = lines[:start_idx] + [replacement] + lines[end_idx:]
    with open("scripts/update_notebook.py", "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Successfully updated update_notebook.py!")
else:
    print(f"Could not find CELL_INFRA_CODE block: {start_idx}, {end_idx}")
