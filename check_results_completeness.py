import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='path to where results csv files are stored')

args = parser.parse_args()

results_path = os.path.abspath(args.results_path)

assert os.path.exists(results_path)

all_files = sorted([f for f in os.listdir(results_path) if f.endswith(".csv")])

for f in all_files:
    df = pd.read_csv(os.path.join(results_path, f))
    
    dataset_name_column = df.columns[0]
    
    df = df[df[dataset_name_column].str.endswith(".pt")]
    
    num_of_dataset_included = df[dataset_name_column].nunique()
    if num_of_dataset_included != 68:
        print(f"merde! for file {f} found {num_of_dataset_included} datasets")
        
    num_of_rows_per_dataset = df.groupby(dataset_name_column)["acc"].count()
    
    datasets_with_too_many_rows = num_of_rows_per_dataset[num_of_rows_per_dataset>9]
    
    if len(datasets_with_too_many_rows) > 0:
        print(f"Found datasets with too many rows in {f}:")
        print(datasets_with_too_many_rows.head())

# print(all_files)
