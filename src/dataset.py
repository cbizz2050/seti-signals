#dataset.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Compose
import pandas as pd
import numpy as np


class SETIDataset(Dataset):
    def __init__(self, root_dir, labels_csv, transform=None):
        self.root_dir = Path(root_dir)
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform
        self.augmentations = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filename = self.labels.iloc[index, 0]
        subfolder = filename[0]
        file_path = os.path.join(self.root_dir, subfolder, filename + ".npy")

        # Load data
        data = np.load(file_path).astype(np.float32)
        data = np.vstack(data).transpose((1, 0))

        # Convert data to PIL Image
        data = Image.fromarray(data)

        if self.augmentations:
            data = self.augmentations(data)

        # Convert PIL Image back to NumPy array
        data = np.array(data)

        # Add a new axis for the single channel (1, height, width)
        data = data[np.newaxis, :, :]

        # Normalize data
        data = (data - data.min()) / (data.max() - data.min())

        # Convert to tensor
        data = torch.from_numpy(data).float()

        label = torch.tensor(self.labels.iloc[index, 1], dtype=torch.long)

        return data, label
