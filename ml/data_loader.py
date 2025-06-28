import pandas as pd
from sklearn.datasets import load_iris
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(dataset_def, data_dir='data', target_column='target'):
    """Load dataset from a definition dictionary.

    Parameters
    ----------
    dataset_def : dict
        Dictionary containing at least a ``path`` key and optional ``label_mapping``.
    data_dir : str, optional
        Directory to save built-in datasets such as ``iris``.
    target_column : str, optional
        Name of the target column for applying ``label_mapping``.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with optional label mapping applied.
    """

    path = dataset_def.get('path')
    if path == 'iris':
        logging.info("Loading Iris dataset...")
        iris = load_iris(as_frame=True)
        df = iris.frame
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, 'iris.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Iris dataset saved to {csv_path}")
    else:
        logging.info(f"Loading dataset from {path}...")
        df = pd.read_csv(path)

    mapping = dataset_def.get('label_mapping') or {}
    if mapping:
        df[target_column] = df[target_column].replace(mapping)

    return df

if __name__ == '__main__':
    example_def = {"path": "iris", "label_mapping": {}}
    df = load_data(example_def)
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
