import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
input_path = root / "dataset.json"
output_path = root / "dataset_multi_turn.json"

text = input_path.read_text(encoding="utf-8")
data = json.loads(text)

# Extract dicts from various possible structures
items = []
for entry in data:
    if isinstance(entry, dict):
        items.append(entry)
    elif isinstance(entry, list):
        # If it's a list of dicts or single-item lists
        for e in entry:
            items.append(e)
    else:
        items.append(entry)

# Group into lists of 3 dicts each (last group may be shorter)
groups = [items[i:i+3] for i in range(0, len(items), 3)]

output_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {output_path} with {len(groups)} groups (total items: {len(items)}).")
