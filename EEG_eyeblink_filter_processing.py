"""Command to run this script from terminal:
conda activate neurotech
cd /Users/chrislbove/Desktop/USC/Neurotech/EEG-EyeBlinks
python EEG_eyeblink_filter_processing.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


BLINK_EVENT_ID = {
    "blink_normal": 1,  # code 0 in label file
    "blink_stim": 2,  # code 1
    "blink_soft": 3,  # code 2
}

BLINK_CODE_TO_EVENT = {
    0: BLINK_EVENT_ID["blink_normal"],
    1: BLINK_EVENT_ID["blink_stim"],
    2: BLINK_EVENT_ID["blink_soft"],
}

FREQ_BANDS = {
    "delta": [0.5, 4],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30],
}

# -------------- STEP 1: LOADING DATA AND PREPROCESSING --------------


def load_csv_eeg_subject_windows(subject_index, window_length=0.5, tmin=0.0, tmax=0.5):
    """
    Load one subject from SXX_data.csv and SXX_labels.csv and return:

    raw    : mne.io.Raw
    events : (n_windows, 3) array (one event per 0.5 s window)
    epochs : mne.Epochs of shape (n_windows, n_channels, window_samples)
    y      : 1D array of binary labels (0 = no blink, 1 = blink in window)
    """

    # Folder where THIS script lives
    BASE_DIR = os.path.dirname(__file__)
    # EEG-IO subfolder next to the script
    DATA_DIR = os.path.join(BASE_DIR, "EEG-IO")

    subj = f"S{subject_index:02d}"
    data_path = os.path.join(DATA_DIR, f"{subj}_data_filtered.csv")
    labels_path = os.path.join(DATA_DIR, f"{subj}_labels.csv")

    # ---------- 1) Read EEG CSV and build RawArray ----------
    # data_df = pd.read_csv(data_path, sep=";")
    data_df = pd.read_csv(data_path, sep=None, engine="python")

    # Sampling frequency from first non-NaN in Sampling Rate column
    # Preset default sampling rate (Hz)
    DEFAULT_SFREQ = 250

    # Try to read from CSV first
    if "Sampling Rate" in data_df.columns:
        sfreq = float(data_df["Sampling Rate"].dropna().iloc[0])
    else:
        print(f"'Sampling Rate' column not found, using default {DEFAULT_SFREQ} Hz")
        sfreq = DEFAULT_SFREQ

    # Use all channels except time and sampling rate
    ch_names = [c for c in data_df.columns if c not in ("Time (s)", "Sampling Rate")]
    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # (n_channels, n_times)
    X = data_df[ch_names].to_numpy().T
    raw = mne.io.RawArray(X, info)

    # Time vector (seconds) from CSV
    time_sec = data_df["Time (s)"].to_numpy()
    recording_end_time = float(time_sec[-1])

    # ---------- 2) Parse label file (same as before) ----------
    corrupt_intervals = []  # list of (start_sec, end_sec)
    blink_times = []  # list of blink middle time (sec)
    blink_codes = []  # list of codes 0/1/2 (we’ll ignore type for binary)

    with open(labels_path, "r") as f:
        # First line: "corrupt, n"
        first_line = f.readline().strip()
        _, n_corrupt_str = [s.strip() for s in first_line.split(",")]
        n_corrupt = int(n_corrupt_str)

        # Next n_corrupt lines: "start,end"
        for _ in range(n_corrupt):
            line = f.readline()
            if not line:
                break
            start_str, end_str = [s.strip() for s in line.split(",")]
            start = float(start_str)
            end = float(end_str)
            # end == -1 means until the end of recording
            if end < 0:
                end = recording_end_time
            corrupt_intervals.append((start, end))

        # Next line should be 'blinks'
        blink_header = f.readline().strip()
        assert blink_header.lower().startswith("blinks")

        # Remaining lines: "<blink_time>, <code>"
        for line in f:
            line = line.strip()
            if not line:
                continue
            t_str, code_str = [s.strip() for s in line.split(",")]
            t = float(t_str)
            code = int(code_str)
            blink_times.append(t)
            blink_codes.append(code)

    blink_times = np.array(blink_times, dtype=float)

    # ---------- 3) Helper: does a window overlap a corrupt interval? ----------
    def window_is_corrupt(t_start, t_end):
        for c_start, c_end in corrupt_intervals:
            # overlap if NOT (window entirely before or entirely after interval)
            if not (t_end <= c_start or t_start >= c_end):
                return True
        return False

    # ---------- 4) Build 0.5 s windows across the recording ----------
    window_starts = np.arange(0, recording_end_time - window_length, window_length)
    events_list = []
    labels_list = []

    for w_start in window_starts:
        w_end = w_start + window_length

        # Skip windows that overlap corrupt intervals
        if window_is_corrupt(w_start, w_end):
            continue

        # Binary label: 1 if any blink falls inside this window, else 0
        has_blink = np.any((blink_times >= w_start) & (blink_times < w_end))
        label = 1 if has_blink else 0

        sample = int(round(w_start * sfreq))  # event at window start
        events_list.append([sample, 0, 1])  # event_id = 1 for all windows
        labels_list.append(label)

    events = np.array(events_list, dtype=int)
    y = np.array(labels_list, dtype=int)

    # ---------- 5) Create windowed epochs ----------
    # event_id just needs to map some name to 1; we’re not using it for labels now
    event_id = {"window": 1}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,  # here tmax should equal window_length
        baseline=None,
        preload=True,
    )

    return raw, events, epochs, y


data_dir = "/Neurotech/EEG-EyeBlinks/EEG-IO"  # adjust if needed


# --------------------------------------------------------------------
# STEP 1.5 — LOAD ALL SUBJECTS & CREATE TRAIN/TEST SPLIT
# --------------------------------------------------------------------

n_subjects = 20

subject_epochs = []
subject_labels = []

for subj_idx in range(n_subjects):  # S00 ... S19
    raw_subj, events_subj, epochs_subj, y_subj = load_csv_eeg_subject_windows(
        subject_index=subj_idx,
        window_length=0.5,
        tmin=0.0,
        tmax=0.5,
    )
    subject_epochs.append(epochs_subj)
    subject_labels.append(y_subj)

# Combine across subjects
epochs_all = mne.concatenate_epochs(subject_epochs)
y_all = np.concatenate(subject_labels)

# Track subject per epoch
subj_per_epoch = np.concatenate(
    [np.full(len(ep), fill_value=i, dtype=int) for i, ep in enumerate(subject_epochs)]
)

# Subject-level train/test split
train_subjects = np.arange(0, 15)
test_subjects = np.arange(15, 20)

train_mask = np.isin(subj_per_epoch, train_subjects)
test_mask = np.isin(subj_per_epoch, test_subjects)

epochs_train = epochs_all[train_mask]
epochs_test = epochs_all[test_mask]
y_train = y_all[train_mask]  # 0 = no blink, 1 = blink
y_test = y_all[test_mask]


# -------------- STEP 2: FEATURE CALCULATION --------------


def eeg_power_band(epochs, freq_bands=FREQ_BANDS):
    """
    Calculate band power features for each epoch.

    Parameters:
    epochs: mne.Epochs
    freq_bands: dict of frequency bands

    Returns:
    X_features: (n_epochs, n_channels * n_bands) array of features
    """
    # Compute PSD between 0.5 and 30 Hz
    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30)
    psds, freqs = spectrum.get_data(
        return_freqs=True
    )  # (n_epochs, n_channels, n_freqs)

    # Normalize PSDs across frequencies for each channel
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []

    # For each frequency band, average PSD in that band and flatten across channels
    for fmin, fmax in freq_bands.values():
        band_mask = (freqs >= fmin) & (freqs < fmax)
        psds_band = psds[:, :, band_mask].mean(axis=-1)  # (n_epochs, n_channels)
        X.append(psds_band.reshape(len(psds), -1))  # (n_epochs, n_channels)

    # Concatenate bands → (n_epochs, n_channels * n_bands)
    return np.concatenate(X, axis=1)


# -------------- STEP 3: BINARY CLASSIFICATION (blink vs no blink) --------------

class_names = ["no_blink", "blink"]
class_codes = [0, 1]

# Random-guess baseline
random_guess_train = {
    name: np.mean(y_train == code) for name, code in zip(class_names, class_codes)
}
print("Random guess probabilities (train):", random_guess_train)

random_guess_test = {
    name: np.mean(y_test == code) for name, code in zip(class_names, class_codes)
}
print("Random guess probabilities (test):", random_guess_test)

pipe = make_pipeline(
    FunctionTransformer(eeg_power_band, validate=False),
    RandomForestClassifier(n_estimators=100, random_state=42),
)

pipe.fit(epochs_train, y_train)

y_pred = pipe.predict(epochs_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc*100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred, labels=class_codes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()
