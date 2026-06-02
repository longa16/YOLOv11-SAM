from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch

# Load SAM predictor if checkpoint exists
sam_predictor = None
sam_ckpt = Path(__file__).parent / "sam_vit_h_4b8939.pth"
if sam_ckpt.exists():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam_model = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
        sam_model.to(device=device)
        sam_predictor = SamPredictor(sam_model)
        sam_status = f"SAM chargé ({device})"
    except Exception as e:
        sam_predictor = None
        sam_status = f"Échec de chargement de SAM: {e}"
else:
    sam_status = "Checkpoint SAM manquant"


def find_model_path():
    for file in Path(__file__).parent.iterdir():
        if file.suffix == ".pt" and not file.name.startswith("sam_"):
            return str(file)
    return None


# Load YOLO model
model_path = find_model_path()
#print("le modèle YOLO chargé est :", model_path)
if model_path is None:
    raise SystemExit("Le modèle entrainé n'a pas été trouvé dans le dossier. Placez votre modèle YOLO ou modifiez le chemin.")
model = YOLO(model_path)


def predict(image: Image.Image, conf: float = 0.25, iou: float = 0.70, use_sam: bool = True, border_margin: float = 0.15, smooth_kernel: int = 3, alpha: float = 0.6):
    if image is None:
        return None

    img_rgb = np.array(image) # on convertit l'image en tableau numpy
    results = model(img_rgb, conf=conf, iou=iou) # on passe notre image au modèle yolo 
    r = results[0]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy() # on crée une copie pour dessiner les masques transparents
    all_contours = []

    # Extraction des boîtes de détection YOLO
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                boxes = np.array(r.boxes.xyxy)
            except Exception:
                boxes = []

        # Récupération des masques YOLO grossiers s'ils existent pour calculer les centres de gravité
        yolo_masks = None
        if hasattr(r, "masks") and r.masks is not None:
            try:
                yolo_masks = r.masks.data.cpu().numpy()
            except Exception:
                try:
                    yolo_masks = np.array(r.masks.data)
                except Exception:
                    yolo_masks = None

        # Filtrage optionnel des bordures métalliques sur les côtés gauche et droit
        if border_margin > 0.0 and len(boxes) > 0:
            width = img_rgb.shape[1]
            left_limit = width * border_margin
            right_limit = width * (1.0 - border_margin)
            
            valid_indices = []
            for i, box in enumerate(boxes):
                x_center_box = (box[0] + box[2]) / 2.0
                if left_limit <= x_center_box <= right_limit:
                    valid_indices.append(i)
            
            boxes = boxes[valid_indices]
            if yolo_masks is not None:
                yolo_masks = yolo_masks[valid_indices]

        # 1. Utilisation de SAM avec guidage par boîte + point
        if use_sam and sam_predictor is not None and len(boxes) > 0: 
            sam_predictor.set_image(img_rgb)
            np.random.seed(42) # Pour garder les mêmes couleurs à chaque frame
            for i, box in enumerate(boxes):
                # Détermination du point central de guidage
                x_center, y_center = None, None
                if yolo_masks is not None and i < len(yolo_masks):
                    mask_h, mask_w = yolo_masks[i].shape
                    y_indices, x_indices = np.where(yolo_masks[i] > 0.5)
                    if len(y_indices) > 0:
                        # Centroid en coordonnées du masque
                        x_center_mask = np.mean(x_indices)
                        y_center_mask = np.mean(y_indices)
                        # Mise à l'échelle vers la résolution originale de l'image
                        x_center = x_center_mask * (img_rgb.shape[1] / mask_w)
                        y_center = y_center_mask * (img_rgb.shape[0] / mask_h)

                if x_center is None or y_center is None:
                    x_center = (box[0] + box[2]) / 2.0
                    y_center = (box[1] + box[3]) / 2.0

                point_coords = np.array([[x_center, y_center]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32) # 1 = point au premier plan

                try:
                    # SAM prédit le masque précis en combinant la boîte et le point central mis à l'échelle
                    masks, _, _ = sam_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=np.array(box),
                        multimask_output=False
                    )
                except Exception:
                    continue

                if masks is None or masks.shape[0] == 0:
                    continue

                mask = masks[0]
                mask_bin = (mask > 0.5).astype("uint8") * 255

                if smooth_kernel > 0:
                    kernel = np.ones((smooth_kernel, smooth_kernel), np.uint8)
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Pour éviter le bruit et les trous internes
                    # On ne garde que le contour ayant la plus grande surface
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100: # Ignore les poussières de moins de 100 pixels
                        all_contours.append(largest_contour)
                        # Génération d'une couleur aléatoire vibrante pour ce caillou
                        color = [int(c) for c in np.random.randint(50, 256, 3)]
                        cv2.drawContours(overlay, [largest_contour], -1, color, thickness=cv2.FILLED)

    # 3. Rendu final avec transparence et bordures nettes dessinées par-dessus
    if len(all_contours) > 0:
        blended = cv2.addWeighted(overlay, float(alpha), img_bgr, 1.0 - float(alpha), 0)
        # Dessine un contour blanc solide autour de chaque caillou pour bien visualiser la séparation
        cv2.drawContours(blended, all_contours, -1, (255, 255, 255), thickness=2)
        out_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)

    # Si aucun masque n'a pu être extrait, tracé YOLO par défaut
    try:
        annotated = r.plot()
        return Image.fromarray(annotated)
    except Exception:
        return Image.fromarray(img_rgb)


# Interface web avec Gradio
def main():
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Image à tester"),
            gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confiance minimale"),
            gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Seuil IOU"),
            gr.Checkbox(value=True, label="Utiliser SAM pour affiner les masques"),
            gr.Slider(0.0, 0.35, value=0.15, step=0.01, label="Marge d'exclusion des bords"),
            gr.Slider(0, 15, value=3, step=1, label="Lissage des masques"),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Alpha overlay"),
        ],
        outputs=gr.Image(type="pil", label="Image annotée"),
        title="Interface d'inférence YOLO + SAM - Granulométrie",
        description=f"Modèle YOLO: {Path(model_path).name} — {sam_status}",
    )

    iface.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()


