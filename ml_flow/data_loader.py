import pandas as pd
from sklearn.datasets import load_iris
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_name='iris', data_dir='data'):
    """
    Loads a dataset. Currently supports 'iris'.
    Saves the raw data to data_dir and returns the dataframe.
    """
    if dataset_name == 'iris':
        logging.info(f"Loading Iris dataset...")
        iris = load_iris(as_frame=True)
        df = iris.frame
        os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists
        csv_path = os.path.join(data_dir, 'iris.csv')
        df.to_csv(csv_path, index=False) # Save raw data
        logging.info(f"Iris dataset saved to {csv_path}")
        return df
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

if __name__ == '__main__':
    df = load_data()
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
