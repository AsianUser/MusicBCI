#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# -------------------------------------------------
# Load CSV
# -------------------------------------------------
def parse_eeg_csv(path: str) -> Tuple[Dict[str, str], pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    header_lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("#"):
                header_lines.append(line.rstrip("\n"))
            else:
                break

    metadata = {}
    for h in header_lines:
        t = h.lstrip("#").strip()
        if "=" in t:
            k, v = t.split("=", 1)
            metadata[k.strip()] = v.strip()

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    return metadata, df


# -------------------------------------------------
# Sampling Rate
# -------------------------------------------------
def infer_fs(metadata, df, time_col="Time"):
    for k, v in metadata.items():
        if "Sample_Frequency" in k:
            return float(re.findall(r"[\d\.]+", v)[0])

    t = df[time_col].values
    return 1.0 / np.median(np.diff(t))


# -------------------------------------------------
# Trigger Detection
# -------------------------------------------------
def detect_trigger_onsets(df, trig_col, time_col):
    s = df[trig_col].fillna(0).astype(float)
    prev = s.shift(1).fillna(0)

    onset_mask = (s != 0) & (prev == 0)

    trigger_times = df.loc[onset_mask, time_col].values
    trigger_values = s[onset_mask].values.astype(int)

    return trigger_times, trigger_values


# -------------------------------------------------
# Plot with Color-Coded Trigger Windows
# -------------------------------------------------
def plot_with_triggers(
    df_seg, time_col, channels, trigger_times, trigger_values, highlight_window=2.0
):

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    t = df_seg[time_col].values

    # Plot EEG
    for ch in channels:
        plt.plot(t, df_seg[ch].values, linewidth=0.8)

    # Color map for triggers 1–8
    trigger_colors = {
        1: "red",
        2: "blue",
        3: "green",
        4: "purple",
        5: "orange",
        6: "brown",
        7: "pink",
        8: "cyan",
    }

    # Shade windows
    for trig_time, trig_val in zip(trigger_times, trigger_values):
        if trig_val in trigger_colors:
            ax.axvspan(
                trig_time,
                trig_time + highlight_window,
                color=trigger_colors[trig_val],
                alpha=0.25,
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title("EEG (First 100s) — 2s After Each Trigger (Color Coded 1–8)")
    plt.xlim(t.min(), t.max())
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------
def run(path):

    SHOW_SECONDS = 100.0
    HIGHLIGHT_WINDOW = 2.0

    metadata, df = parse_eeg_csv(path)

    time_col = "Time"
    trig_col = "Trigger"

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # First 100 seconds
    df_seg = df[df[time_col] <= SHOW_SECONDS].copy()

    # Channels
    preferred = ["LE", "F4", "C4", "P4", "P3", "C3", "F3", "Pz"]
    channels = [c for c in preferred if c in df.columns]

    # Trigger detection
    trigger_times, trigger_values = detect_trigger_onsets(df, trig_col, time_col)

    # Keep only triggers within 100s
    mask = trigger_times <= SHOW_SECONDS
    trigger_times = trigger_times[mask]
    trigger_values = trigger_values[mask]

    plot_with_triggers(
        df_seg,
        time_col,
        channels,
        trigger_times,
        trigger_values,
        highlight_window=HIGHLIGHT_WINDOW,
    )


# -------------------------------------------------
# Edit File Path Here
# -------------------------------------------------
if __name__ == "__main__":
    file_path = r"MusicBCI_musicheadphone_TamaraRicha_PsychoBen_01_raw.csv"
    run(file_path)
