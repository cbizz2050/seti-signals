import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.preprocessing import load_data, preprocess_data
from src.models import CustomModel
from src.utils import get_device

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.train()
    model.to(device)
    scaler = GradScaler()  # Initialize the GradScaler for mixed precision

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use autocast to enable mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale the loss and perform the backward pass using the GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    return model

if __name__ == "__main__":
    # Load and preprocess data
    train_data, test_data, labels = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(train_data, test_data, labels)

    # Create PyTorch Dataset and DataLoader
    train_dataset = CustomDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, criterion, optimizer, and device
    model = CustomModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = get_device()

    # Train the model using mixed-precision
    trained_model = train_model(model, train_dataloader, criterion, optimizer, device)
