"""
Interface Gradio pour tester les modèles OBB entraînés (YOLOv8n-OBB et YOLOv8s-OBB)
sur des images. Permet de comparer visuellement les deux modèles.
"""

import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import cv2

# Chemins vers les meilleurs poids entraînés (les plus récents)
MODELS = {
    "YOLOv8n-OBB": r"C:\Users\loic.ngassa\runs\obb\comparaison_models\train_YOLOv8n-OBB-6\weights\best.pt",
    "YOLOv8s-OBB": r"C:\Users\loic.ngassa\runs\obb\comparaison_models\train_YOLOv8s-OBB-2\weights\best.pt",
}

# Pré-chargement des modèles
loaded_models = {}
for name, path in MODELS.items():
    try:
        loaded_models[name] = YOLO(path)
        print(f"{name} chargé depuis {path}")
    except Exception as e:
        print(f"Erreur lors du chargement de {name}: {e}")


def predict(image: Image.Image, model_name: str, conf: float = 0.25, iou: float = 0.70):
    """Exécute l'inférence OBB sur l'image avec le modèle sélectionné."""
    if image is None:
        return None

    if model_name not in loaded_models:
        return image

    model = loaded_models[model_name]
    img_rgb = np.array(image)

    # Inférence
    results = model(img_rgb, conf=conf, iou=iou)
    r = results[0]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Dessiner les boîtes orientées à partir des 4 coins (r.obb.xyxyxyxy)
    if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
        corners = r.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2) — 4 coins par boîte
        np.random.seed(42)
        for box_corners in corners:
            pts = box_corners.astype(np.int32).reshape((-1, 1, 2))
            color = [int(c) for c in np.random.randint(50, 256, 3)]
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)
    
    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(annotated)


def predict_both(image: Image.Image, conf: float = 0.25, iou: float = 0.70):
    """Exécute l'inférence avec les deux modèles pour comparaison côte à côte."""
    if image is None:
        return None, None

    img1 = predict(image, "YOLOv8n-OBB", conf, iou)
    img2 = predict(image, "YOLOv8s-OBB", conf, iou)
    return img1, img2


# Interface Gradio
with gr.Blocks(title="Test des modèles OBB") as app:
    gr.Markdown("# Test des modèles OBB entraînés — Granulométrie")
    gr.Markdown(f"Modèles disponibles : **{', '.join(loaded_models.keys())}**")

    with gr.Tab("Test individuel"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Image à tester")
                model_choice = gr.Dropdown(
                    choices=list(loaded_models.keys()),
                    value=list(loaded_models.keys())[0] if loaded_models else None,
                    label="Modèle"
                )
                conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confiance minimale")
                iou_slider = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Seuil IOU")
                btn = gr.Button("Lancer l'inférence", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Résultat")

        btn.click(
            fn=predict,
            inputs=[input_image, model_choice, conf_slider, iou_slider],
            outputs=output_image
        )

    with gr.Tab("Comparaison côte à côte"):
        with gr.Row():
            with gr.Column():
                input_image_cmp = gr.Image(type="pil", label="Image à tester")
                conf_slider_cmp = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confiance minimale")
                iou_slider_cmp = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Seuil IOU")
                btn_cmp = gr.Button("Comparer les deux modèles", variant="primary")
        with gr.Row():
            output_n = gr.Image(type="pil", label="YOLOv8n-OBB")
            output_s = gr.Image(type="pil", label="YOLOv8s-OBB")

        btn_cmp.click(
            fn=predict_both,
            inputs=[input_image_cmp, conf_slider_cmp, iou_slider_cmp],
            outputs=[output_n, output_s]
        )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)
