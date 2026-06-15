import os
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import torch



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(all_labels: list, all_preds: list) -> dict:
    accuracy    = accuracy_score(all_labels, all_preds)
    f1_macro    = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    return {
        "accuracy":    accuracy,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
    }


def print_classification_report(all_labels: list, all_preds: list, idx_to_label: dict):
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nRapport de classification :")
    print(report)


def plot_confusion_matrix(
    all_labels: list,
    all_preds: list,
    idx_to_label: dict,
    save_path: str = None,
):
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")

    ax.set_title("Matrice de confusion (normalisée)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Matrice de confusion sauvegardée : {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    val_f1s: list = None,
    save_path: str = None,
):
    epochs = list(range(1, len(train_losses) + 1))
    n_plots = 3 if val_f1s else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle("Courbes d'entraînement", fontsize=15, fontweight="bold")

    axes[0].plot(epochs, train_losses, "o-", label="Train loss", color="#2563EB")
    axes[0].plot(epochs, val_losses,   "o--", label="Val loss",   color="#DC2626")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], "o-",  label="Train acc", color="#2563EB")
    axes[1].plot(epochs, [a * 100 for a in val_accs],   "o--", label="Val acc",   color="#DC2626")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if val_f1s:
        axes[2].plot(epochs, val_f1s, "o-", label="Val F1 (macro)", color="#16A34A")
        axes[2].set_title("F1-score (macro)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1-score")
        axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Courbes d'entraînement sauvegardées : {save_path}")

    plt.show()


def load_label_mapping(json_path: str) -> dict:
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    return device