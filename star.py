import cv2
import numpy as np
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ==========================================
# --- 1. Configuration & Initialisation ---
# ==========================================
IMAGE_PATH = "Users\loic.ngassa\Desktop\Test\img_test.jpg"  # Votre image originale brute

# ROGNAGE AGRESSIF : On coupe très court à droite (1150) et en bas (750)
Y_MIN, Y_MAX = 160, 750
X_MIN, X_MAX = 300, 1150

# Calibration (Exemple : 1 pixel = 0.08 mm)
PIXEL_TO_MM_RATIO = 0.08

# ==========================================
# --- 2. Chargement et Rognage (Crop) ---
# ==========================================
image_full = cv2.imread(IMAGE_PATH)
if image_full is None:
    print(f"ERREUR : Image '{IMAGE_PATH}' introuvable.")
    exit()

# Rognage immédiat
image_cropped = image_full[Y_MIN:Y_MAX, X_MIN:X_MAX]
image_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)

print(f"Taille de l'image (Rognage Agressif) : {image_rgb.shape}")

# ==========================================
# --- 3. Initialisation StarDist ---
# ==========================================
print("Chargement du modèle StarDist...")
model = StarDist2D.from_pretrained('2D_versatile_he')

# ==========================================
# --- 4. Traitement & Prédiction (MODE EXTRÊME) ---
# ==========================================
print("Normalisation de l'image...")
img_normalized = normalize(image_rgb, 1, 99.8, axis=(0,1,2))

print("Prédiction (Seuils Extrêmes : 5% de confiance)...")
# prob_thresh = 0.05 : On accepte tout ce qui ressemble vaguement à un objet
# nms_thresh = 0.10 : On sépare les objets même s'ils se chevauchent énormément
labels, details = model.predict_instances(img_normalized, prob_thresh=0.05, nms_thresh=0.10)

# ==========================================
# --- 5. Extraction des Données ---
# ==========================================
print("Calcul des diamètres...")
final_instances = []
diameters_mm = []

for label_id in np.unique(labels):
    if label_id == 0: continue # Ignorer le fond
    
    mask = (labels == label_id)
    M = cv2.moments(mask.astype(np.uint8))
    
    # On baisse aussi le filtre anti-poussière pour tout capter (50 pixels min)
    if M["m00"] < 50: continue 
    
    area_px = M["m00"]
    area_mm2 = area_px * (PIXEL_TO_MM_RATIO ** 2)
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    
    final_instances.append(mask)
    diameters_mm.append(diameter_mm)

final_count = len(final_instances)
print(f"\n---> STARDIST (Mode Extrême) : {final_count} cailloux détectés ! <---")

# ==========================================
# --- 6. Affichage ---
# ==========================================
def show_anns(instances_list, axis_handle):
    if len(instances_list) == 0: return
    axis_handle.set_autoscale_on(False)
    for mask in instances_list:
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        # Couleurs très vives pour bien voir la bouillie au centre si elle existe
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3): img[:,:,i] = color_mask[i]
        axis_handle.imshow(np.dstack((img, mask*0.7))) # Transparence ajustée à 0.7 pour mieux voir les bords

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(image_rgb)
axes[0].set_title("Image Rognée (Focus 100% Cailloux)", fontsize=14)
axes[0].set_axis_off()

axes[1].imshow(image_rgb)
show_anns(final_instances, axes[1])
axes[1].set_title(f"Segmentation Extrême : {final_count} cailloux", fontsize=16)
axes[1].set_axis_off()

plt.tight_layout()
plt.show()