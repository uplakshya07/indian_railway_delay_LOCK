import pandas as pd
import os

# List of CSV files in the Train delay and prediction dataset[cite: 1]
csv_files = [
    "data_sets/train_details.csv", 
    "data_sets/station_full_names.csv", 
    "data_sets/combined_schedule.csv", 
    "data_sets/combined_delay.csv"
]

def describe_datasets_complete(files):
    for file in files:
        if not os.path.exists(file):
            print(f"Dataset '{file}' not found in the current directory.\n")
            continue
            
        # Load the dataset
        df = pd.read_csv(file, low_memory=False)
        
        # Calculate total number of tuples (rows) for the entire dataset
        total_tuples = len(df)
        
        print(f"=== Dataset: {file} ===")
        print(f"Total number of tuples: {total_tuples}\n")
        
        for column in df.columns:
            # Number of non-empty tuples
            non_empty_count = df[column].notna().sum()
            
            # Number of unique values
            unique_count = df[column].nunique(dropna=True)
            
            # List of unique values and the count of each
            value_counts = df[column].value_counts(dropna=True)
            
            # Truncating the display to top 15 items to handle the ~30 million rows in combined_delay.csv safely[cite: 1]
            if len(value_counts) > 15:
                display_counts = [f"'{val}': {count}" for val, count in value_counts.head(15).items()]
                display_counts.append("... (truncated)")
            else:
                display_counts = [f"'{val}': {count}" for val, count in value_counts.items()]
            
            print(f"Attribute: {column}")
            print(f" - Number of non-empty tuples: {non_empty_count}")
            print(f" - Number of unique values: {unique_count}")
            print(f" - Unique values and their counts (Top 15): [{', '.join(display_counts)}]\n")

describe_datasets_complete(csv_files)