import json

from datasets import load_dataset


def load_and_format_hf_subset(repo="liminghao1630/API-Bank", sample_size=1):
    # 1. Load in streaming mode (no large downloads)
    ds = load_dataset(repo, streaming=True, split="test")

    formatted_subset = []

    print(f"🚀 Streaming from {repo}...")

    # 2. Iterate and transform on the fly
    # We take the first 2000 to shuffle from, or just use shuffle() buffer
    # shuffled_ds = ds.shuffle(buffer_size=1000, seed=42)
    data_list = []
    for i, entry in enumerate(ds):
    #     print(entry)
        data_list.append(entry)
        if i >= sample_size:
            break
        print(entry)
        enrty

    #     # Map ToolBench fields to your schema
    #     # ToolBench typically uses 'query' and 'answer'
    #     print(entry)
    #     query = entry.get("query") or entry.get("instruction", "N/A")

    #     # Determine the type based on presence of API calls
    #     response = entry.get("answer", "")
    #     resp_type = "api_call" if "api" in str(response).lower() else "final_answer"

    #     formatted_subset.append(
    #         {"query": query, "groundtruth": {"type": resp_type, "response": response}}
    #     )

    # # 3. Save to local JSON
    # output_file = "dataset/tool_calling/toolbench_hf_subset.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(formatted_subset, f, indent=4, ensure_ascii=False)

    # print(f"✅ Created {output_file} with {len(formatted_subset)} entries.")


# Run the loader
load_and_format_hf_subset()
