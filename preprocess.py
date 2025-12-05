# EEG-IO/S01_data.csv
# EEG-IO/S01_labels.csv

# channels = ["FP1", "FP2","Channel 3","Channel 4","Channel 5","Channel 6","Channel 7","Channel 8","Channel 9","Channel 10","Channel 11",]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt, resample

# -------- SETTINGS --------
TARGET_SAMPLING_RATE = 250
TIME_COL = "Time (s)"
SAMPLING_RATE_COL = "Sampling Rate"
CHANNELS = [
    "FP1",
    "FP2",
    "Channel 3",
    "Channel 4",
    "Channel 5",
    "Channel 6",
    "Channel 7",
    "Channel 8",
    "Channel 9",
    "Channel 10",
    "Channel 11",
]
HIGH_PASS = 1.0  # Hz
LOW_PASS = 40.0  # Hz
# --------------------------


def load_eeg_csv(path):
    df = pd.read_csv(path, sep=";", engine="python")
    df = df.dropna(axis=1, how="all")
    df[SAMPLING_RATE_COL] = df[SAMPLING_RATE_COL].fillna(method="ffill")
    return df


def notch_filter(signal, fs, freq=60.0):
    b, a = iirnotch(freq / (fs / 2), 30)
    return filtfilt(b, a, signal)


def bandpass_filter(signal, fs, low=HIGH_PASS, high=LOW_PASS):
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def process_eeg(df):
    original_fs = df[SAMPLING_RATE_COL].iloc[0]
    eeg_channels = [ch for ch in CHANNELS if ch in df.columns]
    time = df[TIME_COL].values
    eeg_data = df[eeg_channels].values

    # Resample
    num_samples = int((time[-1] - time[0]) * TARGET_SAMPLING_RATE)
    resampled_data = {}
    for i, ch in enumerate(eeg_channels):
        data = resample(eeg_data[:, i], num_samples)
        data = notch_filter(data, TARGET_SAMPLING_RATE)
        data = bandpass_filter(data, TARGET_SAMPLING_RATE)
        resampled_data[ch] = data

    new_time = np.linspace(time[0], time[-1], num_samples)
    out_df = pd.DataFrame(
        {TIME_COL: new_time, SAMPLING_RATE_COL: TARGET_SAMPLING_RATE, **resampled_data}
    )
    return out_df


def plot_side_by_side(unfiltered_df, filtered_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Unfiltered
    for ch in CHANNELS:
        if ch in unfiltered_df.columns:
            axes[0].plot(unfiltered_df[TIME_COL], unfiltered_df[ch], label=ch)
    axes[0].set_title("Original Unfiltered EEG")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(loc="upper right", ncol=2)

    # Filtered
    for ch in CHANNELS:
        if ch in filtered_df.columns:
            axes[1].plot(filtered_df[TIME_COL], filtered_df[ch], label=ch)
    axes[1].set_title("Filtered & Resampled EEG")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()


def main():
    for i in range(20):  # 0 → 19
        subj = f"S{i:02d}"  # formats to S00, S01, ..., S19

        input_file = f"EEG-IO/{subj}_data.csv"
        output_file = f"EEG-IO/{subj}_data_filtered.csv"

        print(f"\n=== Processing {subj} ===")

        try:
            df = load_eeg_csv(input_file)
        except FileNotFoundError:
            print(f"File not found: {input_file}, skipping...")
            continue

        filtered = process_eeg(df)
        filtered.to_csv(output_file, index=False)

        print(f"Saved filtered file to {output_file}")

        # plot_side_by_side(df, filtered)


if __name__ == "__main__":
    main()
