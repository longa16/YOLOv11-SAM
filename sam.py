import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage import measure

# ==========================================
# 1. PARAMÈTRES À CONFIGURER
# ==========================================
CHEMIN_IMAGE = "img_test.jpg"
CHEMIN_MODELE_SAM = "sam_vit_h_4b8939.pth" # Assurez-vous que le nom correspond au fichier téléchargé
TYPE_MODELE = "vit_h"

# --- CALIBRATION CRITIQUE ---
# Remplacez cette valeur par votre propre calcul après avoir mesuré les 10cm sur l'image
PIXELS_PAR_MM = 8.5  

# ==========================================
# 2. INITIALISATION DE SAM
# ==========================================
print("Chargement du modèle Segment Anything...")
# Utilise la carte graphique si disponible, sinon le processeur classique
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu" 

sam = sam_model_registry[TYPE_MODELE](checkpoint=CHEMIN_MODELE_SAM)
sam.to(device=device)

# Configuration de la sensibilité de l'algorithme
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32, 
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100, # Ignore les poussières
)

# ==========================================
# 3. TRAITEMENT DE L'IMAGE
# ==========================================
print(f"Lecture de l'image {CHEMIN_IMAGE}...")
image = cv2.imread(CHEMIN_IMAGE)

if image is None:
    raise ValueError(f"Impossible de lire l'image. Vérifiez que {CHEMIN_IMAGE} est dans le bon dossier.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Génération des masques par l'IA (Cela peut prendre un moment)...")
masques_generes = mask_generator.generate(image_rgb)
print(f"Terminé ! {len(masques_generes)} cailloux potentiels détectés.")

# ==========================================
# 4. ANALYSE ET MESURES (scikit-image)
# ==========================================
print("Extraction des mesures géométriques...")
donnees_cailloux = []

for idx, mask_data in enumerate(masques_generes):
    masque_binaire = mask_data['segmentation'].astype(np.uint8)
    regions = measure.regionprops(masque_binaire)
    
    if len(regions) > 0:
        region = regions[0]
        
        # Calculs et conversions pixels vers mm
        surface_mm2 = region.area / (PIXELS_PAR_MM ** 2)
        diametre_eq_mm = region.equivalent_diameter_area / PIXELS_PAR_MM
        longueur_mm = region.axis_major_length / PIXELS_PAR_MM
        largeur_mm = region.axis_minor_length / PIXELS_PAR_MM

        donnees_cailloux.append({
            "ID_Caillou": idx + 1,
            "Surface (mm2)": round(surface_mm2, 2),
            "Diametre_Equivalent (mm)": round(diametre_eq_mm, 2),
            "Longueur_Max (mm)": round(longueur_mm, 2),
            "Largeur_Max (mm)": round(largeur_mm, 2)
        })

# ==========================================
# 5. VISUALISATION DES RÉSULTATS
# ==========================================
print("Génération de l'image de visualisation...")
image_masques_couleur = np.zeros_like(image_rgb)
np.random.seed(42) # Fixe les couleurs pour qu'elles restent les mêmes à chaque test

for mask_data in masques_generes:
    masque_binaire = mask_data['segmentation']
    couleur = np.random.randint(0, 255, (3,), dtype=np.uint8)
    image_masques_couleur[masque_binaire] = couleur 

# Superposer les couleurs (40% d'opacité) sur l'image d'origine (60% d'opacité)
image_superposee = cv2.addWeighted(image_rgb, 0.6, image_masques_couleur, 0.4, 0)

# Préparer la fenêtre d'affichage
plt.figure(figsize=(12, 8))
plt.imshow(image_superposee)
plt.title(f"Analyse Granulométrique : {len(masques_generes)} Cailloux Segmentés")
plt.axis('off')

# ==========================================
# 6. EXPORTATION DES DONNÉES
# ==========================================
df_resultats = pd.DataFrame(donnees_cailloux)
fichier_sortie = "resultats_granulometrie.csv"
df_resultats.to_csv(fichier_sortie, index=False, sep=';')

print("\n--- RÉSUMÉ DE L'ANALYSE ---")
print(f"Nombre total de cailloux mesurés : {len(df_resultats)}")
if not df_resultats.empty:
    print(f"Diamètre moyen : {df_resultats['Diametre_Equivalent (mm)'].mean():.2f} mm")
    print(f"Plus gros caillou (longueur max) : {df_resultats['Longueur_Max (mm)'].max():.2f} mm")
print(f"Les données détaillées ont été sauvegardées dans : {fichier_sortie}")

# Afficher l'image à la toute fin (cela va bloquer le script jusqu'à ce que vous fermiez la fenêtre)
print("Affichage de l'image... (Fermez la fenêtre pour terminer le script)")
plt.show()