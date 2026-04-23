import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Helpers ─────────────────────────────────────────────────────────────────


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


# ── Label name maps ──────────────────────────────────────────────────────────
# Edit these to match your actual trigger meanings.
GLOBAL_CLASS_NAMES: dict[int, str] = {
    0: "noActivity",
    1: "blink",
    2: "winkLeft",
    3: "winkRight",
    4: "jawFull",
    5: "jawLeft",
    6: "jawRight",
    7: "faceLeft",
    8: "faceRight",
}


def class_names_for_labels(labels: list[int]) -> list[str]:
    return [GLOBAL_CLASS_NAMES.get(l, str(l)) for l in sorted(labels)]


# ── Main ─────────────────────────────────────────────────────────────────────


def generate_all_confusion_matrices(
    results: dict,  # output of train.main()
    valid_loaders: dict,  # {"full": loader, "trigger_0": loader, ..., "group_eye": loader, ...}
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

        model = result["model"].to(device)
        loader = valid_loaders[key]

        y_true, y_pred = collect_preds(model, loader, device)

        present_labels = sorted(set(y_true) | set(y_pred))
        names = class_names_for_labels(present_labels)

        plot_confusion_matrix(
            y_true,
            y_pred,
            title=f"Confusion Matrix — {key}  (val acc={result['final_val_acc']:.3f})",
            save_path=output_dir / f"cm_{key}.png",
            class_names=names,
        )


# ── Standalone usage ──────────────────────────────────────────────────────────
# If you just want to run this file directly, import your train pipeline here.

if __name__ == "__main__":
    from train import (
        main as run_training,
        build_full_loaders,
        BATCH_SIZE,
        SEED,
        DATASET_KWARGS,
        ROOT_DIR,
    )
    from DataLoader import make_per_trigger_dataloaders, make_three_group_dataloaders
    from torch.utils.data import DataLoader

    # ── Re-run (or load cached) training ─────────────────────────────────────
    print("Running training pipeline …")
    results = run_training()

    # ── Rebuild the validation loaders in the same order ─────────────────────
    # (DataLoaders are not stored in `results`; we rebuild them with the same
    #  seeds so the splits are identical to what the model was validated on.)
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

    # ── Generate ───────────────────────────────────────────────────────────────
    generate_all_confusion_matrices(
        results, valid_loaders, output_dir="confusion_matrices"
    )
