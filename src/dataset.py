import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

class SETIDataset(Dataset):
    def __init__(self, root_dir, labels_csv, transform=None):
        self.root_dir = Path(root_dir)
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        file_id = self.labels.iloc[idx, 0].strip()
        subfolder = file_id[0]  # Get the first character of the file_id as the subfolder
        file_path = self.root_dir / subfolder / f"{file_id}.npy"
        print("file_path:", file_path)
        data = np.load(file_path).astype(np.float32)
        label = self.labels.iloc[idx, 1]

        if self.transform:
            data = self.transform(data)

        return data, label





'''

# Usage example
train_dataset = SETIDataset(root_dir="E:\\seti-breakthrough-listen", labels_csv="E:\\seti-breakthrough-listen\\train_labels.csv")

# Create a DataLoader with multiple workers
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2)

# Iterate through the DataLoader in the training loop
#for batch_idx, (data, labels) in enumerate(train_dataloader):
    # Perform training operations
#    pass
'''