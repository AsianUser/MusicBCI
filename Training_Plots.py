import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Single source of truth — defined in DataLoader.py
from DataLoader import GLOBAL_CLASS_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────


def collect_preds(model, loader, device):
    """Run model over a loader and return (all_labels, all_preds) as numpy arrays."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = torch.argmax(model(inputs), dim=1).cpu()
            all_labels.append(labels)
            all_preds.append(preds)

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path | None = None,
    class_names: list[str] | None = None,
):
    """
    Plot a normalised + raw-count confusion matrix side-by-side.

    Left panel  – normalised (row %) so per-class recall is easy to read.
    Right panel – raw counts for absolute scale.
    """
    labels = sorted(set(y_true) | set(y_pred))

    if class_names is None:
        class_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, matrix, fmt, subtitle in zip(
        axes,
        [cm_norm, cm],
        [".2f", "d"],
        ["Normalised (row %)", "Raw counts"],
    ):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_names,
        )
        disp.plot(ax=ax, colorbar=False, values_format=fmt, cmap="Blues")
        ax.set_title(subtitle)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── Training curves ───────────────────────────────────────────────────────────


def plot_training_curves(
    results: dict,
    output_dir: str = "confusion_matrices",
    ncols: int = 3,
):
    """
    Plot train loss + validation accuracy over epochs for every run in `results`.

    Each subplot covers one model key.  A shared grid is used so all runs are
    easy to compare at a glance.

    Parameters
    ----------
    results    : dict returned by train.main()
    output_dir : folder where the PNG is saved
    ncols      : number of columns in the subplot grid (default 3)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = list(results.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    fig.suptitle(
        "Training curves — loss & validation accuracy", fontsize=15, fontweight="bold"
    )

    # Flatten axes to a 1-D list so we can iterate with zip()
    axes_flat = np.array(axes).flatten()

    for ax, key in zip(axes_flat, keys):
        res = results[key]
        epoch_losses: list[float] = res["epoch_losses"]
        val_ckpts: dict[int, float] = res["val_checkpoints"]  # {epoch: acc}

        epochs_range = range(1, len(epoch_losses) + 1)

        # ── Left y-axis: train loss ───────────────────────────────────────
        color_loss = "#2563EB"  # blue
        ax.plot(
            epochs_range, epoch_losses, color=color_loss, lw=1.5, label="Train loss"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=color_loss)
        ax.tick_params(axis="y", labelcolor=color_loss)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        # ── Right y-axis: validation accuracy ────────────────────────────
        ax2 = ax.twinx()
        color_acc = "#DC2626"  # red
        val_epochs = sorted(val_ckpts.keys())
        val_accs = [val_ckpts[e] for e in val_epochs]

        ax2.plot(
            val_epochs,
            val_accs,
            color=color_acc,
            lw=2,
            marker="o",
            markersize=4,
            label="Val acc",
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Val accuracy", color=color_acc)
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

        # ── Best val-acc marker ───────────────────────────────────────────
        best_epoch = max(val_ckpts, key=val_ckpts.get)
        best_acc = val_ckpts[best_epoch]
        ax2.axvline(best_epoch, color=color_acc, lw=0.8, linestyle="--", alpha=0.5)
        ax2.annotate(
            f"best {best_acc:.1%}",
            xy=(best_epoch, best_acc),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color=color_acc,
        )

        # ── Title & combined legend ───────────────────────────────────────
        best_val_acc = max(val_ckpts.values())
        ax.set_title(f"{key}  (best val acc = {best_val_acc:.3f})", fontsize=9)

        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc="upper right")

    # Hide any unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.show()
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────


def class_names_for_labels(labels: list[int]) -> list[str]:
    return [GLOBAL_CLASS_NAMES.get(l, str(l)) for l in sorted(labels)]


# ── Main ──────────────────────────────────────────────────────────────────────


def generate_all_confusion_matrices(
    results: dict,
    valid_loaders: dict,
    output_dir: str = "confusion_matrices",
):
    """
    Parameters
    ----------
    results       : dict returned by train.main()
    valid_loaders : dict mapping the same keys as `results` → valid DataLoader
    output_dir    : folder where PNGs are saved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)

    for key, result in results.items():
        print(f"\n[{key}]")

        model = result["model"]

        # Load the best checkpoint saved during training
        ckpt_path = Path("checkpoints") / f"model_{result['label']}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"  ↩ Loaded best checkpoint: {ckpt_path}")
        else:
            print(f"  ⚠ No checkpoint found at {ckpt_path}, using final-epoch weights")

        model = model.to(device)
        loader = valid_loaders[key]

        y_true, y_pred = collect_preds(model, loader, device)

        present_labels = sorted(set(y_true) | set(y_pred))
        names = class_names_for_labels(present_labels)

        best_val_acc = max(result["val_checkpoints"].values())

        plot_confusion_matrix(
            y_true,
            y_pred,
            title=f"Confusion Matrix — {key}  (best val acc={best_val_acc:.3f})",
            save_path=output_dir / f"cm_{key}.png",
            class_names=names,
        )


# ── Standalone usage ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    from training_pipe import (
        main as run_training,
        build_full_loaders,
        BATCH_SIZE,
        SEED,
        DATASET_KWARGS,
        ROOT_DIR,
    )
    from DataLoader import (
        make_per_trigger_dataloaders,
        make_three_group_dataloaders,
        make_group_dataloader,
    )
    from torch.utils.data import DataLoader

    # ── Re-run (or load cached) training ──────────────────────────────────
    print("Running training pipeline …")
    results = run_training()

    # ── Rebuild the validation loaders ────────────────────────────────────
    valid_loaders: dict[str, DataLoader] = {}

    # Full dataset
    _, full_valid, _, _ = build_full_loaders(ROOT_DIR)
    valid_loaders["full"] = full_valid

    # Per-trigger
    per_loaders = make_per_trigger_dataloaders(
        ROOT_DIR, batch_size=BATCH_SIZE, train_split=0.8, seed=SEED, **DATASET_KWARGS
    )
    for label, (_, va) in per_loaders.items():
        valid_loaders[f"trigger_{label}"] = va

    # Three groups
    group_loaders = make_three_group_dataloaders(
        ROOT_DIR, batch_size=BATCH_SIZE, train_split=0.8, seed=SEED, **DATASET_KWARGS
    )
    for name, (_, va) in group_loaders.items():
        valid_loaders[name] = va

    _, custom_va = make_group_dataloader(
        ROOT_DIR,
        trigger_labels=[2, 3, 5, 8],
        batch_size=BATCH_SIZE,
        train_split=0.8,
        seed=SEED,
        **DATASET_KWARGS,
    )
    valid_loaders["group_2358"] = custom_va

    # ── Generate ───────────────────────────────────────────────────────────
    generate_all_confusion_matrices(
        results, valid_loaders, output_dir="confusion_matrices"
    )
    plot_training_curves(results, output_dir="confusion_matrices")
