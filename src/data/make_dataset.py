import pandas as pd
import json

# Input and output filenames
excel_file = 'data.xlsx'
json_file = 'knowledge_base.json'

# Read Excel file
df = pd.read_excel(excel_file)

# Convert to desired JSON format
formatted_data = []
for idx, row in df.iterrows():
    formatted_data.append({
        "id": idx + 1,
        "question": str(row['Input']).strip(),
        "answer": str(row['Response']).strip()
    })

# Write to JSON file
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2)

print(f"Converted {excel_file} to {json_file} successfully.")
