import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb

from dataset import load_bbc_dataset
from model import build_model, save_model


# REPRODUCTIBILITE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# TRAIN EPOCH

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # BertForSequenceClassification calcule la loss si on passe les labels
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss   = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# EVAL EPOCH

def eval_epoch(model, loader, device):
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1


# FONCTION PRINCIPALE

def main():
    # Configuration
    SEED       = 42
    DATA_PATH  = "data/bbc-news-data.csv"
    MODEL_NAME = "bert-base-uncased"
    SAVE_PATH  = "best_model"
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    print(f"Device : {DEVICE}")

    # Hyperparametres
    hyperparams = {
        "model_name":  MODEL_NAME,
        "max_length":  256,
        "batch_size":  16,
        "n_epochs":    4,
        "lr":          3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed":        SEED,
    }

    # Chargement du modele et du tokenizer
    model, tokenizer = build_model(num_labels=5, model_name=MODEL_NAME)
    model = model.to(DEVICE)

    # Chargement et preparation du dataset
    train_dataset, val_dataset, label_to_idx, idx_to_label = load_bbc_dataset(
        csv_path   = DATA_PATH,
        tokenizer  = tokenizer,
        max_length = hyperparams["max_length"],
        seed       = SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = hyperparams["batch_size"],
        shuffle    = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = hyperparams["batch_size"],
        shuffle    = False
    )

    # Optimiseur et scheduler
    optimizer = AdamW(
        model.parameters(),
        lr           = hyperparams["lr"],
        weight_decay = hyperparams["weight_decay"]
    )

    total_steps  = len(train_loader) * hyperparams["n_epochs"]
    warmup_steps = int(total_steps * hyperparams["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # Tracking wandb
    wandb.init(
        project = "bert-bbc-news-classification",
        name    = "bert-base-uncased-bbc",
        config  = {**hyperparams, "num_classes": 5, "classes": list(label_to_idx.keys())}
    )

    # Boucle d'entrainement
    best_val_loss = float("inf")

    for epoch in range(hyperparams["n_epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{hyperparams['n_epochs']}")
        print(f"{'='*50}")

        train_loss, train_acc         = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss,   val_acc,  val_f1  = eval_epoch(model, val_loader, DEVICE)

        current_lr = scheduler.get_last_lr()[0]

        print(f"Train  ->  loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%")
        print(f"Val    ->  loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%  f1: {val_f1:.4f}")
        print(f"LR     ->  {current_lr:.2e}")

        wandb.log({
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "train_accuracy": train_acc,
            "val_accuracy":   val_acc,
            "val_f1_score":   val_f1,
            "learning_rate":  current_lr,
            "epoch":          epoch + 1
        })

        # Sauvegarde du meilleur modele selon la val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, tokenizer, SAVE_PATH)
            print(f"Meilleur modele sauvegarde (val_loss={best_val_loss:.4f})")

    print(f"\nEntrainement termine. Meilleure val_loss : {best_val_loss:.4f}")
    wandb.finish()

    # Sauvegarde du mapping des labels pour la demo
    import json
    with open("idx_to_label.json", "w") as f:
        json.dump(idx_to_label, f)
    print("Mapping des labels sauvegarde : idx_to_label.json")


if __name__ == "__main__":
    main()