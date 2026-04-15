import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from WindowMethods import create_windows_raise_triggers, create__windows_generic

from pathlib import Path
from types import SimpleNamespace

# Everyday we shuffling/Splitting
from torch.utils.data import random_split

from WindowMethods import create_windows_generic, create_windows_raise_triggers


SPECIAL_FOLDER = "NOTE MUST Raise All Trigger Vals by 1"


def make_loader(
    csv_path: str, batch_size: int = 8, shuffle: bool = True, **dataset_kwargs
):
    df = pd.read_csv(csv_path, comment="#")

    if SPECIAL_FOLDER in Path(csv_path).parts:
        windows, onsets, labels, eeg_cols = create_windows_raise_triggers(
            df, **dataset_kwargs
        )
    else:
        windows, onsets, labels, eeg_cols = create_windows_generic(df, **dataset_kwargs)

    X = torch.from_numpy(windows).float()  # [N, channels, time]
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    dataset.onsets = onsets
    dataset.eeg_cols = eeg_cols
    dataset.sample_rate = dataset_kwargs.get("sample_rate", 300)
    dataset.chunk_samples = int(
        round(dataset_kwargs.get("chunk_seconds", 1.5) * dataset.sample_rate)
    )
    dataset.skip_samples = int(
        round(
            dataset_kwargs.get("skip_after_prompt_seconds", 0.25) * dataset.sample_rate
        )
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return loader, dataset


def visualize_prompts(dataset, seconds=20, channel_idx=0):

    n_samples = min(int(seconds * dataset.sample_rate), len(dataset.data))
    t = np.arange(n_samples) / dataset.sample_rate

    plt.figure(figsize=(14, 5))
    plt.plot(
        t, dataset.data[:n_samples, channel_idx], label=dataset.eeg_cols[channel_idx]
    )

    for onset in dataset.onsets:
        if onset < n_samples:
            plt.axvline(
                onset / dataset.sample_rate, color="red", linestyle="--", alpha=0.7
            )

    plt.title(f"First {seconds} seconds with prompt markers")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


# To show a full trial instead of the full recording
def visualize_trial(dataset, trial_idx=0, channels_to_plot=None):
    trial_idx = min(trial_idx, len(dataset) - 1)

    onset = dataset.onsets[trial_idx]
    start = onset + dataset.skip_samples
    end = start + dataset.chunk_samples

    x = dataset.data[start:end]  # [time, channels]
    t = np.arange(x.shape[0]) / dataset.sample_rate

    if channels_to_plot is None:
        channels_to_plot = list(range(min(5, x.shape[1])))

    plt.figure(figsize=(14, 6))

    offset = 0.0
    spacing = 3.0
    for ch in channels_to_plot:
        plt.plot(t, x[:, ch] + offset, label=dataset.eeg_cols[ch])
        offset += spacing

    plt.title(f"Trial {trial_idx} | label={dataset.labels[trial_idx]}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + offset")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    ### VISUALIZING

    # visualize_prompts(dataset, seconds=120, channel_idx=0)
    # visualize_trial(dataset, trial_idx=0)
    # CSV file  path (update this to your actual path)
    csv_path = r"MusicBCI_musicheadphone_TamaraRicha_PsychoBen_01_raw.csv"

    loader, dataset = make_loader(
        csv_path,
        batch_size=4,
        sample_rate=300,
        chunk_seconds=1.5,
        skip_after_prompt_seconds=0.25,
        normalize=True,
    )

    gen = torch.Generator().manual_seed(42)

    dataset_size = len(dataset)

    # Splitting into training and testing

    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    valid_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n" * 6)
    print("----- DONE -----")

    # TEST (print first batch shape and labels)
    print("\n" * 6)
    print("-------------- DEBUGGING: First batch shape and labels --------------")
    for X_batch, y_batch in train_loader:
        print("Batch shape:", X_batch.shape)  # [batch, channels, time]
        print("Labels:", y_batch)
        print("Type of X_batch:", type(X_batch))
        print("Type of y_batch:", type(y_batch))
        break
    print("\n" * 6)

    print(valid_loader.dataset)

    print("\n" * 6)


if __name__ == "__main__":
    main()
