import json
import re

# Read source file
with open('/root/autodl-tmp/grpo_format1_200k_3epochs/batch_inference_4models_900_1200_1500_1800/checkpoint-1800-inference/inference_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter entries with <think></think><answer></answer> format
filtered_data = []
for entry in data:
    solution = entry.get('solution', '')
    # Check if solution contains both <think></think> and <answer></answer> tags
    if '<think>' in solution and '</think>' in solution and '<answer>' in solution and '</answer>' in solution:
        filtered_data.append(entry)

print(f"Found {len(filtered_data)} entries with <think></think><answer></answer> format")

# Select first 20 entries
selected_data = filtered_data[:20]

print(f"Selected {len(selected_data)} entries")

# Save to target file
output_path = '/root/autodl-tmp/Agri-R1/Splits/results/selected_20_samples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")

# Print sample for verification
if selected_data:
    print("\n=== Sample entry ===")
    print(f"Question ID: {selected_data[0]['question_id']}")
    print(f"Problem: {selected_data[0]['problem']}")
    print(f"Solution preview: {selected_data[0]['solution'][:200]}...")
