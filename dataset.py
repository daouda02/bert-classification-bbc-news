import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# DATASET PERSONNALISE

class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.

    Prend une liste de textes et de labels, les tokenize avec le tokenizer
    BERT et retourne les tenseurs necessaires pour le modele :
        - input_ids      : indices des tokens dans le vocabulaire BERT
        - attention_mask : 1 pour les vrais tokens, 0 pour le padding
        - label          : indice de la classe (entier long)

    Arguments :
        texts      : liste de textes bruts
        labels     : liste d'indices de classes (entiers)
        tokenizer  : tokenizer Hugging Face (BertTokenizer)
        max_length : longueur maximale de la sequence en tokens (default 256)
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenization avec padding et truncation automatiques
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",  # pad jusqu'a max_length
            truncation      = True,          # tronque si trop long
            return_tensors  = "pt"           # retourne des tenseurs PyTorch
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "label":          torch.tensor(label, dtype=torch.long)
        }



# CHARGEMENT ET PREPARATION DU DATASET BBC NEWS

def load_bbc_dataset(csv_path, tokenizer, max_length=256, test_size=0.2, seed=42):
    df = pd.read_csv(csv_path, sep="\t")

    # Inspection du dataset
    print("Inspection du dataset BBC News")
    print(f"Shape          : {df.shape}")
    print(f"Colonnes       : {df.columns.tolist()}")
    print(f"Valeurs manq.  : {df.isnull().sum().sum()}")
    print(f"\nDistribution des classes :")
    print(df["category"].value_counts())

    # Longueur des textes en mots (approximation avant tokenization)
    df["text_len"] = df["content"].astype(str).apply(lambda x: len(x.split()))
    print(f"\nLongueur des textes (en mots) :")
    print(f"  Min     : {df['text_len'].min()}")
    print(f"  Max     : {df['text_len'].max()}")
    print(f"  Moyenne : {df['text_len'].mean():.0f}")
    print(f"  Mediane : {df['text_len'].median():.0f}")

    print(f"\n5 exemples de textes :")
    for _, row in df.sample(5, random_state=seed).iterrows():
        print(f"  [{row['category']}] {str(row['content'])[:120]}...")

    # Encodage des labels en indices numeriques
    classes      = sorted(df["category"].unique())
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_label = {idx: cls for cls, idx in label_to_idx.items()}

    print(f"\nClasses : {label_to_idx}")

    texts  = df["content"].astype(str).tolist()
    labels = df["category"].map(label_to_idx).tolist()

    # Split 80/20 stratifie pour conserver la distribution des classes
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size    = test_size,
        random_state = seed,
        stratify     = labels
    )

    print(f"\nTrain : {len(X_train)} exemples  |  Val : {len(X_val)} exemples")

    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_length)
    val_dataset   = TextClassificationDataset(X_val,   y_val,   tokenizer, max_length)

    return train_dataset, val_dataset, label_to_idx, idx_to_label