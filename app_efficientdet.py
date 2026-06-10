"""
Entrainement EfficientDet sur dataset COCO (cailloux).

Prerequis:
    pip install effdet timm
"""

import os
import json
import time
import copy
from pathlib import Path
from multiprocessing import freeze_support

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

try:
    from effdet import create_model_from_config, get_efficientdet_config
    from effdet import DetBenchTrain, DetBenchPredict
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = r"C:\Users\loic.ngassa\Downloads\Kalash.coco"
OUTPUT_DIR = r"C:\Users\loic.ngassa\Desktop\YOLOv11-SAM\results_efficientdet"
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.002
IMG_SIZE = 640
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1  # effdet: seulement les classes objets (pas le fond)


# ============================================================
# DATASET COCO
# ============================================================
class CocoDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=640):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size

        ann_file = self.root_dir / "_annotations.coco.json"
        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.root_dir / img_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Redimensionner a la taille cible
        img = img.resize((self.img_size, self.img_size))
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Facteurs d'echelle
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Mettre a l'echelle et convertir en [x1, y1, x2, y2]
            x1 = x * sx
            y1 = y * sy
            x2 = (x + w) * sx
            y2 = (y + h) * sy
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # classe 1 = cailloux

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "bbox": boxes,
            "cls": labels,
            "img_scale": torch.tensor([1.0]),
            "img_size": torch.tensor([self.img_size, self.img_size]),
        }

        return img_tensor, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])

    max_det = max(b[1]["bbox"].shape[0] for b in batch)
    max_det = max(max_det, 1)

    batch_boxes = torch.zeros(len(batch), max_det, 4)
    batch_cls = torch.zeros(len(batch), max_det, dtype=torch.int64)

    for i, (_, t) in enumerate(batch):
        n = t["bbox"].shape[0]
        if n > 0:
            batch_boxes[i, :n] = t["bbox"]
            batch_cls[i, :n] = t["cls"]

    target = {
        "bbox": batch_boxes,
        "cls": batch_cls,
        "img_scale": torch.ones(len(batch)),
        "img_size": torch.tensor([[IMG_SIZE, IMG_SIZE]] * len(batch)),
    }

    return imgs, target


# ============================================================
# MODELE
# ============================================================
def build_model(num_classes):
    """EfficientDet-D0 pre-entraine sur COCO, fine-tune."""
    from omegaconf import OmegaConf
    from effdet.efficientdet import HeadNet

    # 1. Charger avec le nombre de classes original (90) pour que les poids correspondent
    config = get_efficientdet_config("efficientdet_d0")
    config.image_size = [IMG_SIZE, IMG_SIZE]

    net = create_model_from_config(
        config,
        pretrained=True,
        bench_task="train",
    )

    # 2. Deverrouiller la config et mettre a jour num_classes
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)
    config.num_classes = num_classes

    # 3. Mettre a jour la config interne du modele EfficientDet
    OmegaConf.set_readonly(net.model.config, False)
    OmegaConf.set_struct(net.model.config, False)
    net.model.config.num_classes = num_classes

    # 4. Remplacer la tete de classification
    net.model.class_net = HeadNet(config, num_outputs=num_classes)

    # 5. Reconstruire le wrapper DetBenchTrain avec le bon anchor_labeler
    bench = DetBenchTrain(net.model, create_labeler=True)
    return bench


def build_predict_model(train_model, num_classes):
    """Construit le modele en mode inference."""
    config = get_efficientdet_config("efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = [IMG_SIZE, IMG_SIZE]

    net = create_model_from_config(
        config,
        pretrained=False,
        bench_task="predict",
    )
    net.model.load_state_dict(train_model.model.state_dict())
    return net


# ============================================================
# ENTRAINEMENT
# ============================================================
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, targets in data_loader:
        images = images.to(device)
        target = {
            "bbox": targets["bbox"].to(device),
            "cls": targets["cls"].to(device),
            "img_scale": targets["img_scale"].to(device),
            "img_size": targets["img_size"].to(device),
        }

        output = model(images, target)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(predict_model, data_loader, device, conf_threshold=0.25, iou_threshold=0.5):
    predict_model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    inference_times = []

    for images, targets in data_loader:
        images = images.to(device)
        gt_boxes_batch = targets["bbox"]
        gt_cls_batch = targets["cls"]

        start = time.time()
        detections = predict_model(images)
        inference_times.append((time.time() - start) / len(images) * 1000)

        for i in range(len(images)):
            # detections shape: [batch, max_det, 6] -> [x1,y1,x2,y2,score,class]
            dets = detections[i]
            pred_scores = dets[:, 4]
            keep = pred_scores >= conf_threshold
            pred_boxes = dets[keep, :4]

            gt_boxes = gt_boxes_batch[i]
            gt_cls = gt_cls_batch[i]
            valid_gt = gt_cls > 0
            gt_boxes = gt_boxes[valid_gt]

            if len(gt_boxes) == 0:
                all_fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue

            ious = torchvision.ops.box_iou(pred_boxes.cpu(), gt_boxes)
            matched_gt = set()
            for j in range(len(pred_boxes)):
                max_iou, max_k = ious[j].max(0)
                if max_iou >= iou_threshold and max_k.item() not in matched_gt:
                    all_tp += 1
                    matched_gt.add(max_k.item())
                else:
                    all_fp += 1
            all_fn += len(gt_boxes) - len(matched_gt)

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_inference_ms": np.mean(inference_times) if inference_times else 0,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    freeze_support()

    if not EFFDET_AVAILABLE:
        print("ERREUR: le package 'effdet' n'est pas installe.")
        print("Installez-le avec:")
        print("  C:\\Users\\loic.ngassa\\venv_cuda\\Scripts\\pip.exe install effdet timm")
        exit(1)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Datasets
    train_dataset = CocoDetectionDataset(DATASET_ROOT, split="train", img_size=IMG_SIZE)
    val_dataset = CocoDetectionDataset(DATASET_ROOT, split="valid", img_size=IMG_SIZE)
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Modele d'entrainement
    train_model = build_model(NUM_CLASSES)
    train_model.to(DEVICE)

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1 = 0
    best_state = None

    print(f"\n{'='*60}")
    print("  EfficientDet-D0")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(train_model, optimizer, train_loader, DEVICE)
        lr_scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            predict_model = build_predict_model(train_model, NUM_CLASSES)
            predict_model.to(DEVICE)
            metrics = evaluate(predict_model, val_loader, DEVICE)
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} | "
                  f"{metrics['avg_inference_ms']:.1f} ms/img")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = copy.deepcopy(train_model.model.state_dict())
        else:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    # Sauvegarder le meilleur modele
    if best_state is not None:
        save_path = os.path.join(OUTPUT_DIR, "best.pt")
        torch.save(best_state, save_path)
        print(f"\nMeilleur modele sauvegarde: {save_path}")

        # Evaluation finale
        train_model.model.load_state_dict(best_state)
        predict_model = build_predict_model(train_model, NUM_CLASSES)
        predict_model.to(DEVICE)
        final = evaluate(predict_model, val_loader, DEVICE)
        print(f"\nResultats finaux EfficientDet-D0:")
        print(f"  Precision: {final['precision']:.4f}")
        print(f"  Rappel:    {final['recall']:.4f}")
        print(f"  F1-Score:  {final['f1']:.4f}")
        print(f"  Inference: {final['avg_inference_ms']:.1f} ms/image")

    print(f"\nEntrainement termine. Resultats dans: {OUTPUT_DIR}")
