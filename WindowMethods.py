import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def create_windows_generic(
    df: pd.DataFrame,
    sample_rate: int = 300,
    marker_col: str = "Trigger",
    chunk_seconds: float = 1.5,
    skip_after_prompt_seconds: float = 0,
    label_from_trigger: bool = True,
    normalize: bool = True,
):
    exclude = {
        "Time",
        marker_col,
        "Time_Offset",
        "ADC_Status",
        "ADC_Sequence",
        "Event",
        "Comments",
    }
    eeg_cols = [c for c in df.columns if c not in exclude]

    for c in eeg_cols + [marker_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=eeg_cols).reset_index(drop=True)

    data = df[eeg_cols].to_numpy(dtype=np.float32)
    trigger = df[marker_col].fillna(0).astype(int).to_numpy()

    if normalize:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std

    chunk_samples = int(round(chunk_seconds * sample_rate))
    skip_samples = int(round(skip_after_prompt_seconds * sample_rate))

    onsets = np.where((trigger != 0) & np.r_[True, trigger[:-1] == 0])[0]

    windows = []
    valid_onsets = []
    labels = []

    for idx, onset in enumerate(onsets):
        start = onset + skip_samples
        end = start + chunk_samples

        if end > len(data):
            continue

        next_onset = onsets[idx + 1] if idx + 1 < len(onsets) else len(data)
        if end > next_onset:
            continue

        windows.append(data[start:end].T.copy())  # [channels, time]
        valid_onsets.append(int(onset))
        labels.append(int(trigger[onset]) if label_from_trigger else 0)

    if len(windows) == 0:
        return np.empty((0, len(eeg_cols), chunk_samples), dtype=np.float32), [], []

    gap_boundaries = [0] + list(onsets) + [len(data)]

    for gap_start, gap_end in zip(gap_boundaries[:-1], gap_boundaries[1:]):
        region_start = gap_start + skip_samples  # stay clear of the previous trigger
        region_end = gap_end - skip_samples  # stay clear of the upcoming trigger
        win_end = region_start + chunk_samples

        if win_end > region_end:
            continue  # gap too small

        windows.append(data[region_start:win_end].T.copy())
        valid_onsets.append(int(region_start))
        labels.append(0)

    return np.stack(windows).astype(np.float32), valid_onsets, labels, eeg_cols


def create_windows_raise_triggers(
    df: pd.DataFrame,
    sample_rate: int = 300,
    marker_col: str = "Trigger",
    chunk_seconds: float = 1.5,
    skip_after_prompt_seconds: float = 0,
    normalize: bool = True,
    long_gap_seconds: float = 10.0,
):
    exclude = {
        "Time",
        marker_col,
        "Time_Offset",
        "ADC_Status",
        "ADC_Sequence",
        "Event",
        "Comments",
    }
    eeg_cols = [c for c in df.columns if c not in exclude]

    for c in eeg_cols + [marker_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=eeg_cols).reset_index(drop=True)

    data = df[eeg_cols].to_numpy(dtype=np.float32)
    trigger = df[marker_col].fillna(0).astype(int).to_numpy()

    if normalize:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std

    chunk_samples = int(round(chunk_seconds * sample_rate))
    skip_samples = int(round(skip_after_prompt_seconds * sample_rate))
    long_gap_samples = int(round(long_gap_seconds * sample_rate))

    onsets = np.where((trigger != 0) & np.r_[True, trigger[:-1] == 0])[0]

    windows = []
    window_onsets = []
    labels = []

    if len(onsets) == 0:
        return np.empty((0, len(eeg_cols), chunk_samples), dtype=np.float32), [], []

    def has_other_trigger(start: int, end: int, current_onset: int) -> bool:
        other_onsets = onsets[onsets != current_onset]
        return np.any((other_onsets >= start) & (other_onsets < end))

    blocks = []
    current_block = [onsets[0]]

    for prev_onset, curr_onset in zip(onsets[:-1], onsets[1:]):
        if curr_onset - prev_onset > long_gap_samples:
            blocks.append(current_block)
            current_block = [curr_onset]
        else:
            current_block.append(curr_onset)

    blocks.append(current_block)

    for block in blocks:
        for onset in block:
            label = int(trigger[onset]) + 1

            post_start = onset + skip_samples
            post_end = post_start + chunk_samples
            if post_end <= len(data) and not has_other_trigger(
                post_start, post_end, onset
            ):
                windows.append(data[post_start:post_end].T.copy())
                window_onsets.append(int(onset))
                labels.append(label)

            pre_end = onset
            pre_start = pre_end - chunk_samples
            if pre_start >= 0 and not has_other_trigger(pre_start, pre_end, onset):
                windows.append(data[pre_start:pre_end].T.copy())
                window_onsets.append(int(onset))
                labels.append(label)

    if len(windows) == 0:
        return np.empty((0, len(eeg_cols), chunk_samples), dtype=np.float32), [], []

    return np.stack(windows).astype(np.float32), window_onsets, labels, eeg_cols
