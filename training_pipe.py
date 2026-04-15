import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from DataLoader import make_dataset_from_folder
from model import EEG_CNN

root_dir = r"MusicBCI_Data"  # folder containing subfolders and CSVs

dataset, meta = make_dataset_from_folder(
    root_dir,
    sample_rate=300,
    chunk_seconds=1.5,
    skip_after_prompt_seconds=0.25,
    normalize=True,
)

gen = torch.Generator().manual_seed(42)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=gen)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

x_batch, y_batch = next(iter(train_loader))
input_channels = x_batch.shape[1]
num_classes = int(torch.max(dataset.tensors[1]).item()) + 1

model = EEG_CNN(input_channels=input_channels, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/10 | Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Validation accuracy: {correct / total:.4f}")
