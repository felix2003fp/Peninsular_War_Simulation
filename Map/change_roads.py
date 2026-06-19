import pandas as pd

# Configuration
input_file = "edges.csv"
output_file = "edges.csv"
column_name = "edge_id"  # Change to the column you want to modify
start_idx = 322

# Load workbook and select active sheet
df = pd.read_csv(input_file)

# Update values
df.loc[start_idx:, column_name] = [
    f"RD{i}" for i in range(323, 323 + len(df.loc[start_idx:]))
]

# Save modified workbook
df.to_csv(output_file, index=False)

print(f"Updated column {column_name} from row {start_idx} onwards.")
print(f"Saved as {output_file}")