import pytest
import pandas as pd
import numpy as np

def flatten_array(nested_array):
    """
    Flattens a nested array or list into a single list.
    
    Args:
    nested_array (array or list): The nested array to flatten.
    
    Returns:
    list: A flattened list.
    """
    if nested_array is None:
        return []
    return [item for sublist in nested_array for item in sublist]

@pytest.mark.parametrize("file_path", [
    '/data/gpfs/projects/punim2196/kabir/Data/parquet_files/res_leg_press_1.parquet'
])
def test_single_element_in_txt_8k_press(file_path):
    """
    Tests if there is only one element in the 'txt_8k_press' field for all records.
    
    Args:
    file_path (str): Path to the parquet file.
    """
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Convert the dataframe to a list of dictionaries
    records = df.to_dict('records')
        
    # Iterate over each record and assert the condition
    for record in records:
        # Check if 'txt_8k_press' is None
        if record['txt_8k_press'] is None:
            continue
        # Flatten the 'txt_8k_press' array
        flattened_txt_8k_press = flatten_array(record['txt_8k_press'])
        
        if len(flattened_txt_8k_press)>1:
            breakpoint()
        
        # Assert that there is only one element in 'txt_8k_press'
        assert len(flattened_txt_8k_press) == 1, f"Record {record['form_id']} failed: Found {len(flattened_txt_8k_press)} elements in 'txt_8k_press'."
        
        continue

