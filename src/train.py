import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import ViTDetectionModel, EfficientNetModel
from dataset import SETIDataset
from preprocess import load_data, preprocess_data

# Function to sample data
def sample_data(data, labels, sample_percentage):
    sample_size = int(len(data) * sample_percentage)
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    return data[indices], labels[indices]

# Hyperparameters
epochs = 50
batch_size = 32
learning_rate = 1e-4
sample_percentage = 0.1  # Use 10% of the data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
train_labels_path = 'data/train_labels.csv'
train_folder = 'data/train'
test_folder = 'data/test'
train_data, test_data, labels = load_data(train_labels_path, train_folder, test_folder)

# Sample data
train_data, labels = sample_data(train_data, labels, sample_percentage)

# Preprocess data
X_train, X_val, y_train, y_val = preprocess_data(train_data, test_data, labels)

# Create SETI Datasets
train_set = SETIDataset(X_train, y_train)
val_set = SETIDataset(X_val, y_val)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Create models
models = [
    ("ViTDetectionModel_LSTM", ViTDetectionModel(rnn_type="lstm")),
    ("ViTDetectionModel_GRU", ViTDetectionModel(rnn_type="gru")),
    ("EfficientNetModel", EfficientNetModel())
]

for model_name, model in models:
    print(f"Training {model_name}")

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=f"logs/{model_name}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    # Save the trained model
    torch.save(model.state_dict(), f"{model_name}_trained.pth")

# Close the SummaryWriter to release resources
writer.close()
