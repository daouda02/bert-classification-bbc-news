import json
import torch
import torch.nn.functional as F
import gradio as gr
from model import load_model
from utils import load_label_mapping, get_device

MODEL_PATH      = "best_model"
LABEL_MAP_PATH  = "idx_to_label.json"
MAX_LENGTH      = 256
NUM_LABELS      = 5


print("Chargemnt du modèle... ")
device       = get_device()
model, tokenizer = load_model(MODEL_PATH, num_labels=NUM_LABELS)
model        = model.to(device)
model.eval()

idx_to_label = load_label_mapping(LABEL_MAP_PATH)
print(f"Classes détectées : {idx_to_label}")


def predict(text: str) -> dict:
    if not text or not text.strip():
        return {label: 0.0 for label in idx_to_label.values()}

    encoding = tokenizer(
        text,
        max_length     = MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).squeeze(0)  # (num_labels,)

    confidences = {
        idx_to_label[i]: round(probs[i].item(), 4)
        for i in range(NUM_LABELS)
    }

    return confidences

EXAMPLES = [
    [
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
        "signalling further tightening ahead as inflation remains above its 2% target. "
        "Markets fell sharply following the announcement, with the S&P 500 dropping 1.5%."
    ],
    [
        "Manchester United secured a dramatic 2-1 victory over Arsenal at Old Trafford "
        "on Sunday, with a stoppage-time winner from Marcus Rashford. The result moves "
        "United into third place in the Premier League table."
    ],
    [
        "Apple unveiled its latest iPhone at its annual keynote event in Cupertino. "
        "The new model features an upgraded A17 chip, improved camera system with "
        "periscope zoom, and a titanium frame, marking a significant design overhaul."
    ],
    [
        "The Prime Minister announced a sweeping cabinet reshuffle on Monday, replacing "
        "the Home Secretary and Chancellor amid growing pressure from backbenchers. "
        "Opposition leaders called the changes 'too little, too late'."
    ],
    [
        "Oscar-winning director Christopher Nolan's latest film broke box office records "
        "in its opening weekend, grossing over 200 million dollars worldwide. Critics "
        "praised the cinematography and Hans Zimmer's haunting score."
    ],
]

def build_interface() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="BBC News Classifier") as demo:

        gr.Markdown(
            """
            # BBC News : Classificateur de texte
            **Modèle :** BERT (`bert-base-uncased`) fine-tuné sur le dataset BBC News  
            **Classes :** Business · Entertainment · Politics · Sport · Tech

            Saisissez un texte en anglais (titre, article, ou extrait) et le modèle
            prédit sa catégorie avec les probabilités associées.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label       = "Texte à classifier",
                    placeholder = "Entrez un texte en anglais...",
                    lines       = 6,
                )
                submit_btn = gr.Button("Classifier", variant="primary")
                clear_btn  = gr.Button("Effacer", variant="secondary")

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label     = "Prédiction",
                    num_top_classes = NUM_LABELS,
                )

        gr.Examples(
            examples   = EXAMPLES,
            inputs     = text_input,
            label      = "Exemples pré-remplis (cliquez pour charger)",
        )

        gr.Markdown(
            """
            **Note :** Le modèle a été entraîné sur des articles BBC News en anglais.
            Les résultats peuvent être moins précis sur des textes très courts ou hors domaine.
            """
        )

        # ── Interactions ──
        submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)
        text_input.submit(fn=predict, inputs=text_input, outputs=label_output)
        clear_btn.click(fn=lambda: ("", None), outputs=[text_input, label_output])

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        share       = True,   
        server_name = "0.0.0.0",
        server_port = 7860,
    )