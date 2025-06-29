"""
This script provides functionality to load and visualize data from a .npy file.

The primary data file used for demonstration is `748987.npy`, which contains
test data for a partial discharge measurement (idMeasurement, DP19) with a
fault annotation value of 0. The numpy array has dimensions 1x800000.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def load_npy_file(file_path: str) -> Optional[np.ndarray]:
    """
    Loads data from a .npy file.

    This function handles the loading of a numpy array from a specified .npy file.
    It includes error handling for cases where the file cannot be loaded.

    Args:
        file_path: The absolute or relative path to the .npy file.

    Returns:
        A numpy array containing the data from the file, or None if an error occurs.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

def plot_npy_data(data: np.ndarray, file_name: str = "data") -> None:
    """
    Visualizes the data from a numpy array using a line plot.

    Args:
        data: The numpy array to be plotted.
        file_name: A descriptive name for the data source to be used in the plot title.
    """
    if data is not None and data.size > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(data.T)  # Transpose to plot columns if data is 2D
        plt.title(f'Visualization of {file_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    else:
        print("No data available to plot.")


if __name__ == "__main__":
    # Example usage:
    # Path to the numpy file, using a relative path for better portability
    file_path = r"data/748987.npy"
    
    # Load the data
    loaded_data = load_npy_file(file_path)
    
    if loaded_data is not None:
        print(f"Successfully loaded data from {file_path}.")
        print(f"Array shape: {loaded_data.shape}")
        
        # Plot the data
        plot_npy_data(loaded_data, file_name="748987.npy")
    else:
        print(f"Failed to load data from {file_path}.")
