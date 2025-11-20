import numpy as np
import re

def extract_psf_from_content(file_content):
    """
    Extracts the first PSF grid from the file content into a NumPy array.

    Args:
        file_content (str): The full string content of the .int file.

    Returns:
        np.ndarray: A 2D NumPy array of the PSF, or None if an error occurs.
    """
    
    # 1. Find the grid dimensions from the 'GRD' line
    # [cite: 4] shows: "GRD  63   63   FIL  WVL   0.5320 ..."
    grd_match = re.search(r"GRD\s+(\d+)\s+(\d+)", file_content)
    
    if not grd_match:
        print("Error: Could not find 'GRD' line to determine PSF dimensions.")
        return None

    rows = int(grd_match.group(1))
    cols = int(grd_match.group(2))
    total_elements = rows * cols
    
    print(f"Found grid dimensions: {rows}x{cols}")
    print(f"Expecting {total_elements} data points.")

    # 2. Find the start of the numerical data
    # The data starts on the lines immediately following the 'GRD' line.
    data_start_index = grd_match.end()
    data_string = file_content[data_start_index:]

    # skip the column row 
    data_string = data_string[data_string.find('\n'):]

    # 3. Extract all numbers from the rest of the string
    # This finds all sequences of digits, ignoring whitespace/newlines.
    number_strings = re.findall(r"\d+", data_string)

    # 4. Check if we found enough numbers
    if len(number_strings) < total_elements:
        print(f"Error: Found only {len(number_strings)} numbers, but expected {total_elements}.")
        return None

    # 5. Take only the numbers for this PSF and convert to integers
    psf_flat_list = [int(n) for n in number_strings[:total_elements]]
    # breakpoint()
    # 6. Reshape the flat list into the 2D NumPy array
    try:
        psf_array = np.array(psf_flat_list).reshape((rows, cols))
        return psf_array
    except ValueError as e:
        print(f"Error reshaping array: {e}")
        return None