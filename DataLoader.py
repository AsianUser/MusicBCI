import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from WindowMethods import create_windows_generic


def load_one_csv(csv_path: Path, **dataset_kwargs):
    df = pd.read_csv(csv_path, comment="#")
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


def make_per_trigger_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    seed: int = 42,
    **dataset_kwargs,
) -> dict[int, tuple[DataLoader, DataLoader]]:
    """
    Build one (train_loader, valid_loader) pair per unique trigger label.

    Each pair contains ONLY windows for that specific trigger label
    combined with all no-activity windows (label 0), so every loader
    is a clean binary-ish slice of the full dataset.

    Label 0  → no-activity only  (pure baseline loader)
    Label N  → trigger-N windows  +  no-activity windows

    Returns
    -------
    dict  {label: (train_loader, valid_loader)}
        Keys are ints matching the values found in y_all.
    """
    _, meta = make_dataset_from_folder(root_dir, **dataset_kwargs)

    X_all: torch.Tensor = meta["X"]  # [N, channels, time]
    y_all: torch.Tensor = meta["y"]  # [N]

    # unique_labels = sorted(y_all.unique().tolist())
    unique_labels = 9  # sanity cap for printing
    no_activity_mask = y_all == 0

    print(
        f"Found {len(unique_labels)} unique labels: {[int(l) for l in unique_labels]}"
    )
    print(f"No-activity windows (label 0): {no_activity_mask.sum().item()}\n")

    per_trigger_loaders: dict[int, tuple[DataLoader, DataLoader]] = {}

    # Shared generator for DataLoader shuffling (reproducible batch ordering)
    shuffle_gen = torch.Generator().manual_seed(seed)

    for label in unique_labels:
        if label == 0:
            # Baseline loader: all no-activity windows only
            mask = no_activity_mask
        else:
            # Trigger-N loader: that trigger  +  no-activity windows
            trigger_mask = y_all == label
            mask = trigger_mask | no_activity_mask

        X_sub = X_all[mask]
        y_sub = y_all[mask]

        ds = TensorDataset(X_sub, y_sub)

        n_total = len(ds)
        n_train = int(train_split * n_total)
        n_valid = n_total - n_train

        # Each label gets its own seed so splits are independent but reproducible
        split_gen = torch.Generator().manual_seed(seed + int(label))
        train_ds, valid_ds = random_split(ds, [n_train, n_valid], generator=split_gen)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_gen
        )
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

        per_trigger_loaders[int(label)] = (train_loader, valid_loader)

        # Label counts inside this subset
        trigger_count = (y_sub != 0).sum().item() if label != 0 else 0
        baseline_count = (y_sub == 0).sum().item()
        print(
            f"  Label {int(label):>2d} | total={n_total:>5d} "
            f"(train={n_train}, valid={n_valid}) "
            f"[trigger={trigger_count}, baseline={baseline_count}]"
        )

    return per_trigger_loaders


def make_group_dataloader(
    root_dir: str,
    trigger_labels: list[int],
    batch_size: int = 32,
    train_split: float = 0.8,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """
    Build a (train_loader, valid_loader) pair containing windows whose
    label is in `trigger_labels`.  Label 0 (no-activity) is always included
    automatically — no need to add it to the list yourself.

    Parameters
    ----------
    trigger_labels : list[int]
        Trigger labels to keep alongside label-0.
        E.g. [1, 2, 3]  →  keeps labels {0, 1, 2, 3}.

    Returns
    -------
    (train_loader, valid_loader)
    """
    _, meta = make_dataset_from_folder(root_dir, **dataset_kwargs)

    X_all: torch.Tensor = meta["X"]
    y_all: torch.Tensor = meta["y"]

    # Always include label 0; add the requested triggers
    keep = set([0] + [int(l) for l in trigger_labels])
    mask = torch.zeros(len(y_all), dtype=torch.bool)
    for lbl in keep:
        mask |= y_all == lbl

    X_sub = X_all[mask]
    y_sub = y_all[mask]

    ds = TensorDataset(X_sub, y_sub)
    n_total = len(ds)
    n_train = int(train_split * n_total)
    n_valid = n_total - n_train

    split_gen = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(ds, [n_train, n_valid], generator=split_gen)

    shuffle_gen = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_gen
    )
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    counts = {int(l): (y_sub == l).sum().item() for l in sorted(keep)}
    print(
        f"  Group {sorted(keep)} | total={n_total} "
        f"(train={n_train}, valid={n_valid}) | per-label counts: {counts}"
    )

    return train_loader, valid_loader


def make_three_group_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    seed: int = 42,
    **dataset_kwargs,
) -> dict[str, tuple[DataLoader, DataLoader]]:
    """
    Returns three fixed group loaders:
        "group_eye"  → labels {0, 1, 2, 3}
        "group_jaw"  → labels {0, 4, 5, 6}
        "group_face" → labels {0, 7, 8}

    Returns
    -------
    dict  {"group_eye": (train, valid), "group_jaw": ..., "group_face": ...}
    """
    groups = {
        "group_eye": [1, 2, 3],
        "group_jaw": [4, 5, 6],
        "group_face": [7, 8],
    }

    print("\n----- THREE-GROUP DATALOADERS -----")
    loaders: dict[str, tuple[DataLoader, DataLoader]] = {}
    for name, triggers in groups.items():
        loaders[name] = make_group_dataloader(
            root_dir,
            trigger_labels=triggers,
            batch_size=batch_size,
            train_split=train_split,
            seed=seed,
            **dataset_kwargs,
        )

    return loaders


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

    gen = torch.Generator().manual_seed(41526)
    # was 42

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

    # ── Per-trigger loaders ────────────────────────────────────────────────
    print("\n----- PER-TRIGGER DATALOADERS -----")
    per_loaders = make_per_trigger_dataloaders(
        root_dir,
        batch_size=32,
        train_split=0.8,
        seed=42,
        sample_rate=300,
        chunk_seconds=1.5,
        skip_after_prompt_seconds=0.25,
        normalize=True,
    )

    # Quick sanity-check: print one batch from each per-trigger loader
    for label, (tr_loader, va_loader) in per_loaders.items():
        for X_b, y_b in tr_loader:
            print(
                f"\n  [Label {label}] batch shape={X_b.shape}, "
                f"unique y in batch={y_b.unique().tolist()}"
            )
            break

    # ── Three-group loaders ────────────────────────────────────────────────
    group_loaders = make_three_group_dataloaders(
        root_dir,
        batch_size=32,
        train_split=0.8,
        seed=42,
        sample_rate=300,
        chunk_seconds=1.5,
        skip_after_prompt_seconds=0.25,
        normalize=True,
    )

    for group_name, (tr_loader, va_loader) in group_loaders.items():
        for X_b, y_b in tr_loader:
            print(
                f"\n  [{group_name}] batch shape={X_b.shape}, "
                f"unique y in batch={y_b.unique().tolist()}"
            )
            break


if __name__ == "__main__":
    main()
