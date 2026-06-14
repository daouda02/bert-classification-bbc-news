import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer


# CHARGEMENT DU MODELE BERT

def build_model(num_labels, model_name="bert-base-uncased"):
    print(f"Chargement du modele : {model_name}")
    print(f"Nombre de classes    : {num_labels}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model     = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres totaux       : {total:,}")
    print(f"Parametres entrainables : {trainable:,}")

    return model, tokenizer


def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Modele sauvegarde dans : {path}")


def load_model(path, num_labels):
    tokenizer = BertTokenizer.from_pretrained(path)
    model     = BertForSequenceClassification.from_pretrained(
        path,
        num_labels = num_labels
    )
    print(f"Modele charge depuis : {path}")
    return model, tokenizer