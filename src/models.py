# models.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import timm

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("You should implement the forward method in your model!")

class ViTLSTMDetectionModel(BaseModel):
    def __init__(self, pretrained=True, rnn_type="lstm"):
        super(ViTLSTMDetectionModel, self).__init__()

        # Initialize a vision transformer
        vit_config = ViTConfig(image_size=256, patch_size=32, num_channels=1, num_classes=1)
        self.vit = ViTModel(vit_config)
        if pretrained:
            self.vit.init_weights()

        # Initialize an LSTM or GRU for temporal sequence analysis
        rnn_layer = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_layer(input_size=vit_config.hidden_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

        # Final fully connected layers
        self.fc1 = nn.Linear(512 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Process cadence snippets using the vision transformer
        batch_size, sequence_length, _, _ = x.size()
        x = x.view(batch_size * sequence_length, 1, 256, 256)
        x = self.vit(x)
        x = x.pooler_output
        x = x.view(batch_size, sequence_length, -1)

        # Pass the output through the LSTM or GRU
        x, _ = self.rnn(x)

        # Final fully connected layers
        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

class EfficientNetModel(BaseModel):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(EfficientNetModel, self).__init__()

        # Load EfficientNet model
        self.efficientnet = timm.create_model(model_name, pretrained=pretrained, in_chans=1, num_classes=1)

    def forward(self, x):
        # Forward pass through the EfficientNet model
        x = self.efficientnet(x)
        x = torch.sigmoid(x)
        return x

'''
if __name__ == "__main__":
    model1 = ViTLSTMDetectionModel(rnn_type="lstm")
    print("ViTLSTMDetectionModel (LSTM):")
    print(model1)

    model2 = ViTLSTMDetectionModel(rnn_type="gru")
    print("\nViTLSTMDetectionModel (GRU):")
    print(model2)

    model3 = EfficientNetModel()
    print("\nEfficientNetModel:")
    print(model3)
'''