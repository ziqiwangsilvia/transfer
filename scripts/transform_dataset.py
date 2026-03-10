import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
file_path = root / "dataset.json"
backup_path = root / "dataset.json.bak"

text = file_path.read_text(encoding="utf-8")
# Remove triple-backtick fences if present
text = re.sub(r"^\s*```json\s*", "", text, flags=re.IGNORECASE)
text = re.sub(r"```\s*$", "", text)

# Parse JSON
data = json.loads(text)

# If top-level is a list of lists, flatten to list of single-item lists
new_list = []
if isinstance(data, list):
    for item in data:
        if isinstance(item, list):
            for elem in item:
                if isinstance(elem, dict):
                    new_list.append([elem])
                else:
                    new_list.append([elem])
        elif isinstance(item, dict):
            new_list.append([item])
        else:
            new_list.append([item])
else:
    # If it's a single dict, wrap it
    if isinstance(data, dict):
        new_list = [[data]]
    else:
        new_list = [[data]]

# Backup original
backup_path.write_text(text, encoding="utf-8")

# Write transformed JSON
file_path.write_text(json.dumps(new_list, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Transformed {file_path} -> list of {len(new_list)} single-item lists. Backup: {backup_path}")
