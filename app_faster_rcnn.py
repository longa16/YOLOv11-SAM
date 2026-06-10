"""
Entrainement Faster R-CNN (ResNet50-FPN v2) sur dataset COCO .
"""

import os
import json
import time
import copy
from pathlib import Path
from multiprocessing import freeze_support

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = r"C:\Users\loic.ngassa\Downloads\Kalash.coco"
OUTPUT_DIR = r"C:\Users\loic.ngassa\Desktop\YOLOv11-SAM\results_faster_rcnn"
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # fond (0) + cailloux (1)


# ============================================================
# DATASET COCO
# ============================================================
class CocoDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = Path(root_dir) / split
        self.transforms = transforms

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

        anns = self.img_to_anns.get(img_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "area": areas,
            "iscrowd": iscrowd,
        }

        img = T.ToImage()(img)
        img = T.ToDtype(torch.float32, scale=True)(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# MODELE
# ============================================================
def build_model(num_classes):
    """Faster R-CNN ResNet50-FPN v2, pre-entraine sur COCO."""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# ENTRAINEMENT
# ============================================================
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.25):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    inference_times = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        start = time.time()
        outputs = model(images)
        inference_times.append((time.time() - start) / len(images) * 1000)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"][output["scores"] >= conf_threshold]
            gt_boxes = target["boxes"]

            if len(gt_boxes) == 0:
                all_fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue

            ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            for i in range(len(pred_boxes)):
                max_iou, max_j = ious[i].max(0)
                if max_iou >= iou_threshold and max_j.item() not in matched_gt:
                    all_tp += 1
                    matched_gt.add(max_j.item())
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

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Datasets
    train_transforms = T.Compose([T.RandomHorizontalFlip(0.5)])
    train_dataset = CocoDetectionDataset(DATASET_ROOT, split="train", transforms=train_transforms)
    val_dataset = CocoDetectionDataset(DATASET_ROOT, split="valid", transforms=None)
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Modele
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_f1 = 0
    best_state = None

    print(f"\n{'='*60}")
    print("  Faster R-CNN ResNet50-FPN v2")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        lr_scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            metrics = evaluate(model, val_loader, DEVICE)
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} | "
                  f"{metrics['avg_inference_ms']:.1f} ms/img")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())
        else:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    # Sauvegarder le meilleur modele
    if best_state is not None:
        save_path = os.path.join(OUTPUT_DIR, "best.pt")
        torch.save(best_state, save_path)
        print(f"\nMeilleur modele sauvegarde: {save_path}")

        model.load_state_dict(best_state)
        final = evaluate(model, val_loader, DEVICE)
        print(f"\nResultats finaux Faster R-CNN:")
        print(f"  Precision: {final['precision']:.4f}")
        print(f"  Rappel:    {final['recall']:.4f}")
        print(f"  F1-Score:  {final['f1']:.4f}")
        print(f"  Inference: {final['avg_inference_ms']:.1f} ms/image")

    print(f"\nEntrainement termine. Resultats dans: {OUTPUT_DIR}")
