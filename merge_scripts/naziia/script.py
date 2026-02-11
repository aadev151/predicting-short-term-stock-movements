import pandas as pd
import os
from pathlib import Path

src_path = "naziia_data"  # Replace with the actual path where the dataset is downloaded
PROJECT_ROOT = Path.cwd() 
DATA_DIR = PROJECT_ROOT / "base_data"
os.makedirs(DATA_DIR, exist_ok=True)

csv_files = [f for f in os.listdir(src_path) if f.endswith('.csv')]
all_frames = []

print(f"Merging {len(csv_files)} files...")

for file in csv_files:
    file_path = os.path.join(src_path, file)
    # Read the CSV and append to our list
    df = pd.read_csv(file_path)
    all_frames.append(df)

# This creates consolidated NYSE data
merged_df = pd.concat(all_frames, ignore_index=True)
print(f"Done! Combined NYSE data")

# 4. Save to final destination
# output_path = DATA_DIR / "nr-data.csv"
# merged_df.to_csv(output_path, index=False)

# print(f"Done! Combined file saved to: {output_path}")

# 'date' is parsed as a datetime object for better compatibility
df_sp500 = pd.read_csv(DATA_DIR / "sp500_simple.csv", parse_dates=['date'])
merged_df['date'] = pd.to_datetime(merged_df['date'])

final_combined = pd.merge(df_sp500, merged_df, on=['date', 'ticker'], how='outer', suffixes=('_sp500', '_nr'))

output_path = "naziia_output/nr_combined_stock_data.csv"
final_combined.to_csv(output_path, index=False)

print(f"Merge complete! Total rows: {len(merged_df)}")
print(f"File saved to: {output_path}")
