# EEG-IO/S01_data.csv
# EEG-IO/S01_labels.csv

# channels = ["FP1", "FP2","Channel 3","Channel 4","Channel 5","Channel 6","Channel 7","Channel 8","Channel 9","Channel 10","Channel 11",]

import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS --------
eeg_csv_path = "EEG-IO/S01_data.csv"  # your EEG file
blink_csv_path = "EEG-IO/S01_labels.csv"  # your blink events file

channels_to_plot = []  # [] = all channels
time_col = "Time (s)"
sampling_rate_col = "Sampling Rate"
delimiter = ";"
# --------------------------


def load_eeg_csv(path, delimiter=";"):
    df = pd.read_csv(path, sep=delimiter, engine="python")
    df = df.dropna(axis=1, how="all")
    df = df.apply(pd.to_numeric, errors="ignore")

    if sampling_rate_col in df.columns:
        df[sampling_rate_col] = df[sampling_rate_col].fillna(method="ffill")

    return df


def load_blink_csv(path):
    """
    Reads blink events only from lines AFTER the line containing "blinks".
    """
    blink_times = []

    with open(path, "r") as f:
        lines = f.readlines()

    # Find the index of the line containing "blinks"
    start_index = None
    for i, line in enumerate(lines):
        if "blinks" in line.lower():
            start_index = i + 1  # start AFTER this line
            break

    if start_index is None:
        raise ValueError("No 'blinks' marker found in blink CSV.")

    # Parse only the lines after "blinks"
    for line in lines[start_index:]:
        parts = line.strip().split(",")
        try:
            blink_times.append(float(parts[0]))
        except:
            pass

    return blink_times


def plot_channels_with_blinks(df, channels_to_plot, blink_times):
    # Detect EEG channels
    all_channels = [
        col for col in df.columns if col not in [time_col, sampling_rate_col]
    ]

    if len(channels_to_plot) == 0:
        channels = all_channels
    else:
        channels = [ch for ch in channels_to_plot if ch in all_channels]

    plt.figure(figsize=(14, 8))

    # Plot EEG
    for ch in channels:
        plt.plot(df[time_col], df[ch], label=ch)

    # Overlay blink events
    for t in blink_times:
        plt.axvline(x=t, linestyle="--", alpha=0.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("EEG Channels with Blink Event Overlay")
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()


# -------- MAIN --------
df = load_eeg_csv(eeg_csv_path)
blink_times = load_blink_csv(blink_csv_path)

print("Blink events loaded after 'blinks':")
print(blink_times)

plot_channels_with_blinks(df, channels_to_plot, blink_times)
