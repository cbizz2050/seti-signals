# train.py
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.model_selection import train_test_split

from dataset import SETIDataset  # Assuming you have a SETIDataset class defined in seti_dataset.py
from models import EfficientNetModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 100
learning_rate = 1e-4
batch_size = 8  # Reduced batch size to mitigate GPU memory issue
image_size = 224
image_means = [0.485, 0.456, 0.406]
image_stds = [0.229, 0.224, 0.225]
gradient_accumulation_steps = 4  # Gradient accumulation steps

# Transformations
transform = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=image_means, std=image_stds)
])

# Load the dataset
root_dir = "E:\seti-breakthrough-listen\data\\train"
labels_csv = "E:\seti-breakthrough-listen\data\\train_labels.csv"

dataset = SETIDataset(root_dir=root_dir, labels_csv=labels_csv, transform=transform)  # Create the dataset object

# Split the dataset into train and validation sets
train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels.iloc[:, 1])

train_dataset = SETIDataset(root_dir=root_dir, labels_csv=labels_csv, transform=transform)
val_dataset = SETIDataset(root_dir=root_dir, labels_csv=labels_csv, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=4, prefetch_factor=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=4, prefetch_factor=2)

# Mixed-precision training (automatic mixed precision - AMP)
scaler = torch.cuda.amp.GradScaler()

# Create models
models = [
    ("EfficientNetModel_b0", EfficientNetModel(model_name="efficientnet_b0")),
    ("EfficientNetModel_b1", EfficientNetModel(model_name="efficientnet_b1")),
    ("EfficientNetModel_b2", EfficientNetModel(model_name="efficientnet_b2"))
]

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), "best_model.pth")
        self.val_loss_min = val_loss

# ModelCheckpoint class
class ModelCheckpoint:
    def __init__(self, filepath, save_best_only=False):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_val_loss = float("inf")

    def __call__(self, val_loss, model):
        if self.save_best_only:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(model.state_dict(), self.filepath)
        else:
            torch.save(model.state_dict(), self.filepath)

def train_models():
    for model_name, model in models:
        print(f"Training {model_name}")

        # Send model to device
        model.to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, verbose=True)

        # Early stopping and model checkpoint
        early_stopping = EarlyStopping(patience=7, verbose=True)
        model_checkpoint = ModelCheckpoint(f"{model_name}_best.pth", save_best_only=True)

        # TensorBoard summary writer
        writer = SummaryWriter(log_dir=f"runs/{model_name}")

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")
            model.train()

            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()  # Add this line

                # Debugging print statements
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                # Backward pass with mixed-precision training and gradient accumulation
                scaler.scale(loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # Validation loop
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)
                    labels = labels.unsqueeze(1).float()  # Add this line

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_losses.append(loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            scheduler.step(val_loss)
            print(f"Validation loss: {val_loss:.6f}")

            # TensorBoard logging
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            # Early stopping and model checkpoint
            early_stopping(val_loss, model)
            model_checkpoint(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load best model and save
        model.load_state_dict(torch.load("best_model.pth"))
        torch.save(model.state_dict(), f"{model_name}_best.pth")

        # Close TensorBoard writer
        writer.close()