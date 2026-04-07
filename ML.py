'''Assumptions for model input:
- X shape: (n_windows, 7, time_steps)
- y shape: (n_windows,)
- all windows same length
- same sampling rate for all data
- same channel order in every window
- one label per window
- labels standardized across dataset
- windows are mostly single-class, not mixed'''


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_channels=7, n_classes=3):  # ADJUST CLASSES (should be 7?)
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (batch, channels, time_steps)
        x = self.features(x)      # (batch, 32, 1)
        x = x.squeeze(-1)         # (batch, 32)
        x = self.classifier(x)    # (batch, n_classes)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * X_batch.size(0)
        total_correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    all_preds = []
    all_true = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * X_batch.size(0)
        total_correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_batch.cpu().numpy())

    return total_loss / total, total_correct / total, np.array(all_true), np.array(all_preds)


# -------------------------
# Example main script
# -------------------------
X = np.load("X_windows.npy")   # shape should be (samples, channels, time_steps)
y = np.load("y_labels.npy")    # integer labels like (0-clench etc.)

# If X is (samples, time_steps, channels), fix it:
# X = np.transpose(X, (0, 2, 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

n_samples, n_channels, time_steps = X.shape
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = EEGDataset(X_train, y_train)
test_ds = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleEEGCNN(n_channels=n_channels, n_classes=n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred))


'''
you recorded 10 minutes of EEG
you cut it into 1-second windows
sampling rate = n Hz
Then:
each window = n time points
suppose you get 600 windows total
Your data becomes:
X.shape = (600, 7, n)
y.shape = (600,)
So:
600 samples → 600 windows
7 channels → your headset
n time_steps → 1 second of data
'''