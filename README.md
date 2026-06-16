# BERT Classification — BBC News

Fine-tuning de BERT pour la classification d'articles BBC News en 5 catégories.  
Devoir Pratique n°3 — NLP avec PyTorch | Master / Ingénierie IA

---

## Membres du binôme

| Membre | Rôle |
|--------|------|
| Daouda Sow | `dataset.py`, `model.py`, `train.py` — pipeline d'entraînement |
| Seydou Kabore | `utils.py`, `demo.py`, `README.md` — évaluation & démo Gradio |

---

## Dataset

**Source :** BBC News Dataset (5 catégories, articles en anglais)

| Classe | Description |
|--------|-------------|
| business | Économie, finance, marchés |
| entertainment | Cinéma, musique, célébrités |
| politics | Politique nationale et internationale |
| sport | Football, tennis, sports divers |
| tech | Technologie, informatique, sciences |

**Statistiques :**
- ~2 225 articles au total
- Split 80/20 stratifié : ~1 780 train / ~445 validation
- Longueur moyenne : ~380 mots par article
- `max_length = 256` tokens (couvre la majorité des articles)

---

## Modèle

**Architecture :** `bert-base-uncased` (Hugging Face) + tête de classification linéaire (768 → 5)  
**Paramètres totaux :** ~109 M  
**Paramètres entraînables :** ~109 M (fine-tuning complet)

---

## Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 3e-5 |
| Batch size | 16 |
| Epochs | 4 |
| Max length | 256 |
| Optimiseur | AdamW (weight_decay=0.01) |
| Scheduler | Linéaire avec warmup (10%) |
| Loss | CrossEntropyLoss |
| Seed | 42 |

---

## Installation

```bash
git clone https://github.com/daouda02/bert-classification-bbc-news.git
cd bert-classification-bbc-news
pip install -r requirements.txt
```

> **Note NumPy :** si vous avez NumPy 2.x installé, downgrade nécessaire :
> ```bash
> pip install "numpy<2"
> ```

---

## Utilisation

### Entraînement

Placez le dataset dans `data/bbc-news-data.csv` puis :

```bash
python train.py
```

Le meilleur modèle est sauvegardé automatiquement dans `best_model/`.

### Démo interactive

```bash
python demo.py
```

Ouvre `http://localhost:7860` dans votre navigateur.  
Sur Google Colab, un lien public est généré automatiquement (`share=True`).

---

## Structure du projet

```
bert-classification-bbc-news/
├── data/
│   └── bbc-news-data.csv
├── best_model/          # modèle sauvegardé après entraînement
├── dataset.py           # TextClassificationDataset + chargement BBC News
├── model.py             # chargement / sauvegarde du modèle BERT
├── train.py             # boucles train_epoch / eval_epoch + main
├── utils.py             # métriques, visualisations, utilitaires
├── demo.py              # interface Gradio
├── idx_to_label.json    # mapping (classes)
├── requirements.txt
└── README.md
```

---

## Résultats

VOIR curves.png

| Métrique | Valeur |
|----------|--------|
| Val Accuracy | 97.98% 97.98% 98.43% 98.88% |
| Val F1 (macro) | 0.9796 0.9800 0.9845 0.9888 |
| Meilleure val loss | 0.1070 0.0869 0.0889 0.0709 |

---

## Répartition du travail

**Daouda Sow** — pipeline complet d'entraînement : analyse du dataset, tokenisation, `TextClassificationDataset`, fine-tuning BERT, sauvegarde du meilleur modèle.

**Seydou Bécaye Kaboré** — évaluation et déploiement : métriques et visualisations, interface Gradio, documentation.
