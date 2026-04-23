"""
eeg_live.py
===========
Runs the EEG model on live headset data via mne_lsl and writes the predicted
action integer to a shared file for spotify_controller.py to pick up.

Start order:
    1.  python spotify_controller.py
    2.  python eeg_live.py
"""

import torch
import time
import numpy as np
from pathlib import Path
from mne_lsl.stream import StreamLSL

from model import EEG_CNN
from DataLoader import make_dataset_from_folder

# ── Config ────────────────────────────────────────────────────────────────────
SHARED_FILE = Path("/tmp/eeg_action.txt")
COOLDOWN_SECONDS = 1.5
CONFIDENCE_THRESHOLD = 0.6

SAMPLE_RATE = 300
CHUNK_SECONDS = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)  # 450 samples per window

# ── Load model ────────────────────────────────────────────────────────────────
ROOT_DIR = "MusicBCI_Data"
DATASET_KWARGS = dict(
    sample_rate=SAMPLE_RATE,
    chunk_seconds=CHUNK_SECONDS,
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
SHARED_FILE.write_text("0")

# ── Connect to LSL stream ─────────────────────────────────────────────────────
print("Connecting to LSL stream ...")
from mne_lsl.lsl import resolve_streams

# Find all streams
streams = resolve_streams()
print(f"Found {len(streams)} stream(s):")
for stream in streams:
    print(f"  - {stream.name} ({stream.stype}) @ {stream.sfreq} Hz")

# Or filter by name/type
eeg_streams = resolve_streams(timeout=5.0, stype="EEG")
print(f"\nFound {len(eeg_streams)} EEG stream(s)")

dsi_stream = resolve_streams(timeout=5.0, name="DSI-7")
if dsi_stream:
    print(f"\nFound DSI-7 stream: {dsi_stream[0].name}")

stream = StreamLSL(
    bufsize=CHUNK_SECONDS * 2, name=dsi_stream[0].name if dsi_stream else None
)  # name=None picks first available
stream.connect()
print(f"Connected | channels={stream.n_channels} sfreq={stream.info['sfreq']}")


def get_eeg_window() -> np.ndarray | None:
    """
    Pull the latest samples from the LSL buffer.
    Returns a [channels, WINDOW_SAMPLES] numpy array, or None if not enough
    samples have accumulated yet.
    """
    data, _ = stream.get_data(winsize=CHUNK_SECONDS)  # [channels, samples]

    if data.shape[1] < WINDOW_SAMPLES:
        return None  # buffer not full yet — skip this cycle

    # Take the most recent WINDOW_SAMPLES
    return data[:, -WINDOW_SAMPLES:].astype(np.float32)


def predict(window: np.ndarray) -> int:
    """Returns predicted label int (0 = no activity, 1-8 = action)."""
    x = torch.from_numpy(window).float().unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    confidence = probs.max().item()
    label = probs.argmax().item()

    return label if confidence >= CONFIDENCE_THRESHOLD else 0


# ── Live loop ─────────────────────────────────────────────────────────────────
print("Streaming — Ctrl-C to stop\n")
last_action_time = 0.0

try:
    while True:
        window = get_eeg_window()

        if window is None:
            time.sleep(0.01)  # wait for buffer to fill
            continue

        label = predict(window)

        if label != 0:
            now = time.time()
            if (now - last_action_time) >= COOLDOWN_SECONDS:
                SHARED_FILE.write_text(str(label))
                print(f"  -> action {label}")
                last_action_time = now
        else:
            SHARED_FILE.write_text("0")

        time.sleep(CHUNK_SECONDS)  # wait one full window before next prediction

finally:
    stream.disconnect()
    print("LSL stream disconnected.")
