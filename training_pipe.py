import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import random_split, DataLoader

from DataLoader import (
    make_dataset_from_folder,
    make_per_trigger_dataloaders,
    make_three_group_dataloaders,
)
from model import EEG_CNN


# ── Config ─────────────────────────────────────────────────────────────────
ROOT_DIR = r"MusicBCI_Data"
SAMPLE_RATE = 300
CHUNK_SECONDS = 1.5
SKIP_SECONDS = 0.25
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
SEED = 42
VALIDATE_EVERY = 10

DATASET_KWARGS = dict(
    sample_rate=SAMPLE_RATE,
    chunk_seconds=CHUNK_SECONDS,
    skip_after_prompt_seconds=SKIP_SECONDS,
    normalize=True,
)

Path("checkpoints").mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Core training function ──────────────────────────────────────────────────
def train_and_evaluate(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    input_channels: int,
    num_classes: int,
    label: str = "full",
    epochs: int = EPOCHS,
    lr: float = LR,
    validate_every: int = VALIDATE_EVERY,
) -> dict:
    """
    Train a fresh EEG_CNN on one (train_loader, valid_loader) pair.

    Returns
    -------
    dict with keys:
        label           – name/id of this run
        epoch_losses    – list of mean train loss per epoch
        val_checkpoints – dict {epoch: accuracy} for every validation step
        final_val_acc   – accuracy at the last validation checkpoint
        model           – the trained nn.Module (on CPU)
    """
    model = EEG_CNN(input_channels=input_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    val_checkpoints = {}

    print(f"\n{'═' * 55}")
    print(
        f"  Training: [{label}]  |  classes={num_classes}  |  "
        f"train={len(train_loader.dataset)}  valid={len(valid_loader.dataset)}"
    )
    print(f"{'═' * 55}")

    best_val_acc = 0.0  # add before the epoch loop

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mean_loss = running_loss / len(train_loader)
        epoch_losses.append(mean_loss)
        print(f"  Epoch {epoch + 1:>3}/{epochs} | loss: {mean_loss:.4f}")

        # ── Validate ───────────────────────────────────────────────────────
        if (epoch + 1) % validate_every == 0:
            model.eval()
            correct = total = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = torch.argmax(model(inputs), dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            val_checkpoints[epoch + 1] = acc

            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), f"checkpoints/model_{label}.pt")
                print(
                    f"  ✔ Validation @ epoch {epoch + 1}: acc={acc:.4f}  ← new best, saved"
                )
            else:
                print(f"  ✔ Validation @ epoch {epoch + 1}: acc={acc:.4f}")

    final_val_acc = val_checkpoints.get(epochs, list(val_checkpoints.values())[-1])

    return {
        "label": label,
        "epoch_losses": epoch_losses,
        "val_checkpoints": val_checkpoints,
        "final_val_acc": final_val_acc,
        "model": model.cpu(),
    }


# ── Build loaders ───────────────────────────────────────────────────────────
def build_full_loaders(root_dir: str) -> tuple[DataLoader, DataLoader, int, int]:
    """Returns train_loader, valid_loader, input_channels, num_classes."""
    dataset, _ = make_dataset_from_folder(root_dir, **DATASET_KWARGS)

    gen = torch.Generator().manual_seed(SEED)
    n = len(dataset)
    n_train = int(0.8 * n)
    train_ds, test_ds = random_split(dataset, [n_train, n - n_train], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    x_sample, _ = next(iter(train_loader))
    input_channels = x_sample.shape[1]
    num_classes = int(dataset.tensors[1].max().item()) + 1

    print(f"Full dataset  | total={n} | train={n_train} | test={n - n_train}")
    return train_loader, valid_loader, input_channels, num_classes


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> dict[str, dict]:
    """
    Runs one training job per loader and returns all results keyed by name.

    Results dict shape:
        {
          "full":      { label, epoch_losses, val_checkpoints, final_val_acc, model },
          "trigger_0": { ... },
          "trigger_1": { ... },
          ...
        }
    """
    all_results: dict[str, dict] = {}

    # ── 1. Full-dataset run ─────────────────────────────────────────────────
    train_loader, valid_loader, input_channels, num_classes = build_full_loaders(
        ROOT_DIR
    )

    all_results["full"] = train_and_evaluate(
        train_loader,
        valid_loader,
        input_channels=input_channels,
        num_classes=num_classes,
        label="full dataset",
    )

    # ── 2. Per-trigger runs ─────────────────────────────────────────────────
    per_loaders = make_per_trigger_dataloaders(
        ROOT_DIR,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        seed=SEED,
        **DATASET_KWARGS,
    )

    for trigger_label, (tr_loader, va_loader) in per_loaders.items():
        # Each per-trigger dataset is binary-ish: trigger vs baseline (label 0)
        # num_classes must cover the highest label index present in this subset
        subset_y = tr_loader.dataset.dataset.tensors[1]
        nc = int(subset_y.max().item()) + 1

        run_key = f"trigger_{trigger_label}"
        all_results[run_key] = train_and_evaluate(
            tr_loader,
            va_loader,
            input_channels=input_channels,
            num_classes=nc,
            label=run_key,
        )

    # ── 3. Three-group runs ─────────────────────────────────────────────────
    group_loaders = make_three_group_dataloaders(
        ROOT_DIR,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        seed=SEED,
        **DATASET_KWARGS,
    )

    for group_name, (tr_loader, va_loader) in group_loaders.items():
        subset_y = tr_loader.dataset.dataset.tensors[1]
        nc = int(subset_y.max().item()) + 1

        all_results[group_name] = train_and_evaluate(
            tr_loader,
            va_loader,
            input_channels=input_channels,
            num_classes=nc,
            label=group_name,
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print("  SUMMARY — Final validation accuracy")
    print(f"{'─' * 55}")
    for key, res in all_results.items():
        print(f"  {key:<20s}  →  {res['final_val_acc']:.4f}")
    print(f"{'═' * 55}")

    return all_results


if __name__ == "__main__":
    results = main()
