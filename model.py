import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# import torchvision
# import torchvision.transforms as transforms
# import timm

import torch.nn.functional as F
import matplotlib.pyplot as plt  # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm


# class EEGDataset(Dataset):
#     def __init__(self, eeg_data, labels, transform=None):
#         """
#         Args:
#             eeg_data (np.array): A numpy array of EEG data (e.g., shape: [N_samples, N_channels, N_timepoints]).
#             labels (np.array): A numpy array of labels (e.g., shape: [N_samples]).
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.eeg_data = torch.from_numpy(eeg_data).float()
#         self.labels = torch.from_numpy(labels).long() # Use .long() for classification labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         data = self.eeg_data[idx]
#         label = self.labels[idx]

#         if self.transform:
#             data = self.transform(data)

#         return data, label


class EEG_CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEG_CNN, self).__init__()

        # Define 1D convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)

        # Instead of guessing time dimension (like 128),
        # we use adaptive pooling to make it work for ANY window size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # After adaptive pooling, output is [batch, 128, 1] → flatten → 128
        self.fc_input_features = 128

        # Define fully connected layers
        self.fc1 = nn.Linear(self.fc_input_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: [batch, channels, time]

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reduce time dimension to 1
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # IMPORTANT:
        # Do NOT use log_softmax if using CrossEntropyLoss
        return x


# Example of initializing the model
# INPUT_CHANNELS = 32 # number of EEG channels
# NUM_CLASSES = 5   # number of distinct labels
# model = EEG_CNN(INPUT_CHANNELS, NUM_CLASSES)
# import torch.optim as optim

# def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(dataloader):
#             # Move data to GPU if available
#             # inputs, labels = inputs.to(device), labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f}')

# Example of training setup
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_model(model, dataloader, criterion, optimizer)
