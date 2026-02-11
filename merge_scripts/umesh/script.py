import pandas as pd
from pathlib import Path
import os
 
def merge_data():
    project_root = Path(__file__).resolve().parent.parent.parent
    my_data_dir = project_root / "merge_scripts" / "umesh" / "my_data"
    jp_script_data_dir = project_root / "base_data"
    output_dir = project_root / "umesh_merged"
    output_path = output_dir / "sp500_combined.csv"
 
    companies_file = my_data_dir / "sp500_companies.csv"
    index_file = my_data_dir / "sp500_index.csv"
    simple_stocks_file = jp_script_data_dir / "sp500_simple.csv"
 
    print("Loading datasets...")
   
    for f in [companies_file, index_file, simple_stocks_file]:
        if not f.exists():
            print(f"Error: {f} not found.")
            return
 
    print(f"Reading {simple_stocks_file.name}...")
    df_stocks = pd.read_csv(simple_stocks_file)
   
    print(f"Reading {companies_file.name}...")
    df_companies = pd.read_csv(companies_file)
   
    print(f"Reading {index_file.name}...")
    df_index = pd.read_csv(index_file)
 
    print("Merging company metadata...")
    df_combined = pd.merge(df_stocks, df_companies, left_on="ticker", right_on="Symbol", how="left")
    df_combined.drop(columns=["Symbol"], inplace=True)
 
    print("Merging index data...")
    df_combined = pd.merge(df_combined, df_index, left_on="date", right_on="Date", how="left")
    df_combined.drop(columns=["Date"], inplace=True)
 
    print(f"Saving combined table to {output_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
   
    print(f"Successfully created combined table with {len(df_combined)} rows.")
    print(f"Columns: {list(df_combined.columns)}")
 
if __name__ == "__main__":
    merge_data()
