import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from WindowMethods import create_windows_generic, create_windows_raise_triggers

SPECIAL_FOLDER = "NOTE MUST Raise All Trigger Vals by 1"


def load_one_csv(csv_path: Path, **dataset_kwargs):
    df = pd.read_csv(csv_path, comment="#")

    if SPECIAL_FOLDER in csv_path.parts:
        windows, onsets, labels, eeg_cols = create_windows_raise_triggers(
            df, **dataset_kwargs
        )
    else:
        windows, onsets, labels, eeg_cols = create_windows_generic(df, **dataset_kwargs)

    return windows, onsets, labels, eeg_cols


def make_dataset_from_folder(root_dir: str, **dataset_kwargs):
    root_dir = Path(root_dir)
    csv_files = sorted(root_dir.rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found under: {root_dir}")

    all_X = []
    all_y = []
    all_sources = []
    file_meta = []

    for csv_path in csv_files:
        windows, onsets, labels, eeg_cols = load_one_csv(csv_path, **dataset_kwargs)

        if windows is None or len(windows) == 0:
            continue

        X = torch.from_numpy(windows).float()
        y = torch.tensor(labels, dtype=torch.long)

        all_X.append(X)
        all_y.append(y)
        all_sources.extend([str(csv_path)] * len(labels))

        file_meta.append(
            {
                "path": str(csv_path),
                "onsets": onsets,
                "eeg_cols": eeg_cols,
                "n_windows": len(labels),
                "special": SPECIAL_FOLDER in csv_path.parts,
            }
        )

    if not all_X:
        raise ValueError(f"No usable windows found under: {root_dir}")

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    dataset = TensorDataset(X_all, y_all)

    meta = {
        "X": X_all,
        "y": y_all,
        "sources": all_sources,
        "file_meta": file_meta,
        "sample_rate": dataset_kwargs.get("sample_rate", 300),
        "chunk_seconds": dataset_kwargs.get("chunk_seconds", 1.5),
        "skip_after_prompt_seconds": dataset_kwargs.get(
            "skip_after_prompt_seconds", 0.25
        ),
    }

    return dataset, meta


def visualize_trial(meta, trial_idx=0, channels_to_plot=None):
    X = meta["X"]
    y = meta["y"]
    sources = meta["sources"]
    sample_rate = meta["sample_rate"]

    trial_idx = min(trial_idx, len(X) - 1)

    x = X[trial_idx].cpu().numpy()  # [channels, time]
    source = sources[trial_idx]
    t = np.arange(x.shape[1]) / sample_rate

    if channels_to_plot is None:
        channels_to_plot = list(range(min(5, x.shape[0])))

    plt.figure(figsize=(14, 6))

    offset = 0.0
    spacing = 3.0
    for ch in channels_to_plot:
        plt.plot(t, x[ch] + offset, label=f"ch {ch}")
        offset += spacing

    plt.title(
        f"Trial {trial_idx} | label={y[trial_idx].item()} | source={Path(source).name}"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + offset")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    root_dir = r"MusicBCI"

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

    print("----- DONE -----")
    print("Train size:", len(train_ds))
    print("Valid size:", len(test_ds))

    for X_batch, y_batch in train_loader:
        print("Batch shape:", X_batch.shape)
        print("Labels:", y_batch)
        print("Type of X_batch:", type(X_batch))
        print("Type of y_batch:", type(y_batch))
        break

    print(valid_loader.dataset)


if __name__ == "__main__":
    main()
