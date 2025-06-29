"""
This script analyzes the inferred_annotation.csv file to count the number of occurrences
for each unique idStation. It uses the pandas library for efficient data manipulation
and provides a clear summary of the distribution of measurements across stations.
"""

import pandas as pd
from typing import Counter

def count_station_occurrences(file_path: str) -> Counter:
    """
    Reads a CSV file and counts the occurrences of each unique value in the 'idStation' column.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A Counter object with the counts of each unique idStation.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check if 'idStation' column exists
        if 'idStation' not in df.columns:
            print("Error: 'idStation' column not found in the CSV file.")
            return Counter()

        # Count the occurrences of each unique idStation
        station_counts = df['idStation'].value_counts()

        return station_counts

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return Counter()
    except Exception as e:
        print(f"An error occurred: {e}")
        return Counter()

if __name__ == "__main__":
    # Path to the CSV file
    file_path = r"../unitest/data/inferred_annotation.csv"

    # Get the station counts
    station_counts = count_station_occurrences(file_path)

    # Print the results
    if not station_counts.empty:
        print("Number of occurrences for each unique idStation:")
        print(station_counts)
    else:
        print("Could not retrieve station counts.")
