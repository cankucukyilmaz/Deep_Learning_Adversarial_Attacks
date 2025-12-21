import pandas as pd

# 'delim_whitespace=True' handles the spaces between columns.
# 'skiprows=1' skips the very first line which is just the number "202599".
df = pd.read_csv('data/external/list_attr_celeba.txt', delim_whitespace=True, skiprows=1)

# Check the balance
# Replace -1 (False) with 0 for easier calculation
df.replace(-1, 0, inplace=True)

# Calculate the percentage of "1s" (Positives) for each attribute
# The index (filenames) is automatically handled by pandas in this format
balance = df.mean(numeric_only=True) * 100

# Display the results sorted from most rare to most common
print("--- CelebA Attribute Balance (%) ---")
pd.set_option('display.max_rows', None) # Ensure we see all 40 rows
print(balance.sort_values())