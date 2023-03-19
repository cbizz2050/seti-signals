import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def normalize_data(data):
    """
    Normalize the data to the range [0, 1] for each sample.
    """
    return np.array([(sample - sample.min()) / (sample.max() - sample.min()) for sample in data])

def load_data(train_labels_path, train_folder, test_folder):
    """
    Load training and testing data from the given file paths.
    """
    train_labels = pd.read_csv(train_labels_path)
    train_ids = train_labels['id'].values
    
    train_data = []
    for train_id in train_ids:
        file_path = os.path.join(train_folder, f"{train_id}.npy")
        if not os.path.exists(file_path):
            file_path = os.path.join(train_folder, train_id[0], f"{train_id}.npy")
        train_data.append(np.load(file_path))
    
    test_ids = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.endswith(".npy"):
                test_ids.append(file.split('.')[0])

    test_data = []
    for test_id in test_ids:
        file_path = os.path.join(test_folder, f"{test_id}.npy")
        if not os.path.exists(file_path):
            file_path = os.path.join(test_folder, test_id[0], f"{test_id}.npy")
        test_data.append(np.load(file_path))

    return np.array(train_data), np.array(test_data), train_labels['target'].values

def preprocess_data(train_data, test_data, labels, train_labels_path, train_folder, test_folder, test_size=0.2, random_state=42):
    """
    Preprocess the data: normalize and perform a train-validation split.
    """
    # Load data
    train_data, test_data, labels = load_data(train_labels_path, train_folder, test_folder)

    # Normalize data
    train_data_normalized = normalize_data(train_data)
    test_data_normalized = normalize_data(test_data)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(train_data_normalized, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    return X_train, X_val, y_train, y_val, test_data_normalized

if __name__ == "__main__":
    train_labels_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train_labels.csv'
    train_folder = sys.argv[2] if len(sys.argv) > 2 else 'data/train'
    test_folder = sys.argv[3] if len(sys.argv) > 3 else 'data/test'

    # Preprocess data
    X_train, X_val, y_train, y_val, test_data_normalized = preprocess_data(train_data, test_data, labels, train_labels_path, train_folder, test_folder)
