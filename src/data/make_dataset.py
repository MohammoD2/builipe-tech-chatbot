import pandas as pd
import json

# Load the Excel file
df = pd.read_excel("data.xlsx")

# Convert to desired JSON format
services = []
for _, row in df.iterrows():
    services.append({
        "title": str(row["Input"]).strip(),
        "description": str(row["Response"]).strip()
    })

# Save to JSON
with open("services.json", "w", encoding="utf-8") as f:
    json.dump(services, f, indent=2, ensure_ascii=False)

print("✅ Converted Excel to services.json")


