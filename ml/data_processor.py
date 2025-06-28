import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df, features=None, target=None, scaling_method='standard', test_size=0.2, val_size=0.2, random_state=42):
    """
    Preprocesses the  feature scaling and train-val-test split.
    """
    logging.info("Preprocessing data...")

    if features is None:
        features = df.columns.drop(target)
    if target is None:
        target = df.columns[-1] # Assume last column is target if not specified

    X = df[features]
    y = df[target]

    # Feature Scaling
    if scaling_method == 'standard':
        scaler = StandardScaler()
        logging.info("Applying StandardScaler...")
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        logging.info("Applying MinMaxScaler...")
    else:
        raise ValueError(f"Scaling method '{scaling_method}' not supported.")

    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features) # Convert back to DataFrame

    # Train-Val-Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df, y, test_size=(test_size + val_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)

    logging.info("Data preprocessing complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    from .data_loader import load_data
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target='target') # Assuming 'target' column exists after loading
    print("Data preprocessed and split.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
