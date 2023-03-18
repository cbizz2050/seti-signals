import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_data(data):
    """
    Normalize the data to the range [0, 1].
    """
    return (data - data.min()) / (data.max() - data.min())

def load_data(train_labels_path, train_folder, test_folder):
    """
    Load training and testing data from the given file paths.
    """
    train_labels = pd.read_csv(train_labels_path)
    train_ids = train_labels['id'].values
    train_data = [np.load(f"{train_folder}/{train_id}.npy") for train_id in train_ids]

    test_ids = pd.read_csv(test_folder)['id'].values
    test_data = [np.load(f"{test_folder}/{test_id}.npy") for test_id in test_ids]
    
    return np.array(train_data), np.array(test_data), train_labels['target'].values

def preprocess_data(train_data, test_data, labels, test_size=0.2, random_state=42):
    """
    Preprocess the data: normalize and perform a train-validation split.
    """
    # Normalize data
    train_data_normalized = normalize_data(train_data)
    test_data_normalized = normalize_data(test_data)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(train_data_normalized, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    # Load data
    train_labels_path = 'data/train_labels.csv'
    train_folder = 'data/train'
    test_folder = 'data/test'
    train_data, test_data, labels = load_data(train_labels_path, train_folder, test_folder)

    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess_data(train_data, test_data, labels)
