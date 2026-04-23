"""
eeg_live.py
===========
Runs the EEG model on live headset data and writes the predicted
action integer to a shared file for spotify_listener.py to pick up.

Start order:
    1.  python spotify_listener.py   (reads the shared file)
    2.  python eeg_live.py           (writes the shared file)
"""

import torch
import time
import numpy as np
from pathlib import Path

from model import EEG_CNN
from DataLoader import make_dataset_from_folder

# ── Shared state ──────────────────────────────────────────────────────────────
SHARED_FILE = Path("/tmp/eeg_action.txt")
COOLDOWN_SECONDS = 1.5
CONFIDENCE_THRESHOLD = 0.6

# ── Load model ────────────────────────────────────────────────────────────────
ROOT_DIR = "MusicBCI_Data"
DATASET_KWARGS = dict(
    sample_rate=300,
    chunk_seconds=1.5,
    skip_after_prompt_seconds=0.25,
    normalize=True,
)

dataset, meta = make_dataset_from_folder(ROOT_DIR, **DATASET_KWARGS)
input_channels = meta["X"].shape[1]
num_classes = int(dataset.tensors[1].max().item()) + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_CNN(input_channels=input_channels, num_classes=num_classes)
model.load_state_dict(torch.load("checkpoints/model_full.pt", map_location="cpu"))
model.to(device).eval()

print(f"Model loaded | writing actions to {SHARED_FILE}")
SHARED_FILE.write_text("0")  # initialise to no-action


def get_eeg_window() -> np.ndarray:
    """
    Replace this with your real headset SDK call.
    Should return a numpy array of shape [channels, time].
    """
    raise NotImplementedError("Plug your headset SDK in here")


def predict(window: np.ndarray) -> int:
    """Returns predicted label int (0 = no activity, 1–8 = action)."""
    x = torch.from_numpy(window).float().unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    confidence = probs.max().item()
    label = probs.argmax().item()

    return label if confidence >= CONFIDENCE_THRESHOLD else 0


# ── Live loop ─────────────────────────────────────────────────────────────────
last_action_time = 0.0

while True:
    window = get_eeg_window()
    label = predict(window)

    if label != 0:
        now = time.time()
        if (now - last_action_time) >= COOLDOWN_SECONDS:
            SHARED_FILE.write_text(str(label))
            print(f"  → wrote action {label}")
            last_action_time = now
    else:
        SHARED_FILE.write_text("0")
