from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO

try:
    import cv2
except Exception:
    cv2 = None

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

# Load SAM predictor if checkpoint exists
sam_predictor = None
sam_status = "SAM non disponible"
if SAM_AVAILABLE:
    sam_ckpt = Path(__file__).parent / "sam_vit_h_4b8939.pth"
    if sam_ckpt.exists():
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        try:
            sam_model = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
            sam_model.to(device=device)
            sam_predictor = SamPredictor(sam_model)
            sam_status = f"SAM chargé ({device})"
        except Exception:
            sam_predictor = None
            sam_status = "SAM installé mais échec de chargement"
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


def predict(image: Image.Image, conf: float = 0.25, iou: float = 0.5, smooth: bool = True, alpha: float = 0.6, use_sam: bool = True):
    if image is None:
        return None

    img_rgb = np.array(image)
    results = model(img_rgb, conf=conf, iou=iou)
    r = results[0]

    # if SAM is enabled and available, refine masks using boxes from YOLO
    if use_sam and sam_predictor is not None and hasattr(r, "boxes") and len(r.boxes) > 0:
        sam_predictor.set_image(img_rgb)
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                boxes = np.array(r.boxes.xyxy)
            except Exception:
                boxes = []

        if cv2 is not None and len(boxes) > 0:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            overlay = img_bgr.copy()

            for box in boxes:
                try:
                    masks, _, _ = sam_predictor.predict(box=np.array(box), multimask_output=False)
                except Exception:
                    continue

                if masks is None or masks.shape[0] == 0:
                    continue

                mask = masks[0]
                mask_bin = (mask > 0.5).astype("uint8") * 255

                if smooth:
                    kernel = np.ones((3, 3), np.uint8)
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

            blended = cv2.addWeighted(overlay, float(alpha), img_bgr, 1.0 - float(alpha), 0)
            out_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            return Image.fromarray(out_rgb)

    # Fallback: use YOLO mask output if available
    if hasattr(r, "masks") and r.masks is not None:
        try:
            mask_data = r.masks.data.cpu().numpy()
        except Exception:
            mask_data = np.array(r.masks.data)

        if cv2 is None:
            try:
                annotated = r.plot()
                return Image.fromarray(annotated)
            except Exception:
                return Image.fromarray(img_rgb)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()

        for mask in mask_data:
            if mask.ndim == 3:
                mask = mask[0]
            mask_bin = (mask > 0.5).astype("uint8") * 255

            if smooth:
                kernel = np.ones((3, 3), np.uint8)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

        blended = cv2.addWeighted(overlay, float(alpha), img_bgr, 1.0 - float(alpha), 0)
        out_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)

    try:
        annotated = r.plot()
        return Image.fromarray(annotated)
    except Exception:
        return Image.fromarray(img_rgb)


def main():
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Image à tester"),
            gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence"),
            gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IOU"),
            gr.Checkbox(value=True, label="Utiliser SAM pour affiner les masques"),
            gr.Checkbox(value=True, label="Lisser les masques"),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Alpha overlay"),
        ],
        outputs=gr.Image(type="pil", label="Image annotée"),
        title="Interface d'inférence YOLO + SAM",
        description=f"Modèle YOLO: {Path(model_path).name} — {sam_status}",
    )

    iface.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()


def main():
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Image à tester"),
            gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence"),
            gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IOU"),
            gr.Checkbox(value=True, label="Lisser les masques"),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Alpha overlay"),
        ],
        outputs=gr.Image(type="pil", label="Image annotée"),
        title="Interface d'inférence YOLO",
        description=f"Modèle chargé: {Path(model_path).name}",
    )

    iface.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
