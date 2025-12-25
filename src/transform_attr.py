import pandas as pd
import os

def process_attributes(input_path, output_path):
    # Check if file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find file at {input_path}")

    print(f"Reading attributes from {input_path}...")
    
    # Read the file. 
    # CelebA txt files are whitespace separated. 
    # skiprows=2 skips the total count line and the header line usually found in standard CelebA
    # We load the header names separately to ensure alignment
    
    # Read header row (line 2, 0-indexed is 1)
    with open(input_path, 'r') as f:
        _ = f.readline() # Skip count
        header = f.readline().strip().split()
        
    # Read data
    df = pd.read_csv(
        input_path, 
        delim_whitespace=True, 
        header=None, 
        skiprows=2,
        names=['image_id'] + header,
        index_col='image_id' # Set image_id as index
    )

    print("Converting -1 labels to 0...")
    # Apply transformation: -1 becomes 0, 1 stays 1
    # We can use a simple lambda or map, but replacing is efficient
    df.replace(-1, 0, inplace=True)

    # Save to processed folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved processed labels to {output_path}")
    return df

if __name__ == "__main__":
    # Adjust paths based on your screenshot structure
    INPUT_FILE = 'data/external/list_attr_celeba.txt'
    OUTPUT_FILE = 'data/processed/celeba_attrs_clean.csv'
    
    process_attributes(INPUT_FILE, OUTPUT_FILE)