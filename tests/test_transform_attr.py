import pytest
import pandas as pd
import numpy as np
import os

# 1. Define the path constant
PROCESSED_FILE_PATH = 'data/processed/celeba_attrs_clean.csv'

# 2. Create a fixture (optional but cleaner) or just use the constant
@pytest.fixture
def processed_data():
    if not os.path.exists(PROCESSED_FILE_PATH):
        pytest.fail(f"File not found: {PROCESSED_FILE_PATH}. Run the processing script first.")
    return pd.read_csv(PROCESSED_FILE_PATH, index_col=0)

# 3. The test function uses the fixture
def test_data_integrity(processed_data):
    df = processed_data
    
    # Check 1: Values are strictly binary {0, 1}
    unique_values = np.unique(df.values)
    assert np.all(np.isin(unique_values, [0, 1])), \
        f"Failed: Found values {unique_values}. Expected {{0, 1}}."
    
    # Check 2: CelebA has 40 attributes
    assert df.shape[1] == 40, \
        f"Failed: Expected 40 columns, got {df.shape[1]}."

    # Check 3: Check for NaNs
    assert not df.isnull().values.any(), "Failed: Dataset contains NaNs."