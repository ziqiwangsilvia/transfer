import json
from pathlib import Path

# Load the full dataset
dataset_path = Path("synthetic_financial_dataset_20260302.json")
with open(dataset_path) as f:
    data = json.load(f)

# Create subset with 3 entries per category
subset = {}
for key, entries in data.items():
    if isinstance(entries, list):
        subset[key] = entries[:3]  # Take first 3 entries
    else:
        subset[key] = entries

# Save the subset
output_path = Path("synthetic_financial_dataset_subset.json")
with open(output_path, "w") as f:
    json.dump(subset, f, indent=2)

# Print summary
print(f"Original dataset:")
for key, entries in data.items():
    if isinstance(entries, list):
        print(f"  {key}: {len(entries)} entries")

print(f"\nSubset dataset (3 entries per category):")
for key, entries in subset.items():
    if isinstance(entries, list):
        print(f"  {key}: {len(entries)} entries")

print(f"\nSubset saved to: {output_path}")
