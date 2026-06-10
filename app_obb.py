import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from ultralytics import YOLO

# 1. CONFIGURATION
# Remplacez par le chemin absolu ou relatif vers votre fichier data.yaml
DATA_YAML_PATH = r"C:\Users\loic.ngassa\Downloads\Kalash.yolov8-obb\data.yaml"
EPOCHS = 50          # Nombre d'époques pour l'entraînement
IMG_SIZE = 640       # Taille des images
BATCH_SIZE = 16      # Taille du batch (à ajuster selon votre carte graphique)

# Liste des modèles à comparer (du plus léger au plus performant)
MODELS_TO_COMPARE = {
    "YOLOv8n-OBB": "yolov8n-obb.pt",
    "YOLOv8s-OBB": "yolov8s-obb.pt"
}


if __name__ == '__main__':
    freeze_support()

    results_summary = []

    # 2. ENTRAÎNEMENT ET ÉVALUATION
    for model_name, model_weights in MODELS_TO_COMPARE.items():
        print(f"\n" + "="*50)
        print(f"Début du traitement pour : {model_name}")
        print(f"="*50)
        
        # Charger le modèle pré-entraîné
        model = YOLO(model_weights)
        
        # A. Entraînement
        print(f"[{model_name}] Phase d'entraînement...")
        train_results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name=f"train_{model_name}",
            project=r"C:\Users\loic.ngassa\Desktop\YOLOv11-SAM\comparaison_models",
            exist_ok=True,
            workers=0,
        )
        
        # B. Évaluation (Validation/Test)
        print(f"[{model_name}] Phase d'évaluation...")
        metrics = model.val(data=DATA_YAML_PATH, split="val") # ou split="test" selon vos besoins
        
        # Extraction des métriques clés (Spécifiques au format OBB)
        # model.val() retourne un OBBMetrics dont les résultats sont dans .box
        map50 = metrics.box.map50        # mAP à Seuil IoU 0.50
        map50_95 = metrics.box.map       # mAP global (0.50-0.95)
        precision = metrics.box.mp       # Précision Moyenne
        recall = metrics.box.mr          # Rappel Moyen
        fitness = metrics.fitness        # Métrique globale de performance Ultralytics
        
        # Récupérer le temps d'inférence moyen par image (en millisecondes)
        speed_inference = metrics.speed.get('inference', 0)
        
        # Enregistrement des résultats
        results_summary.append({
            "Modèle": model_name,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "Précision": precision,
            "Rappel": recall,
            "Vitesse Inférence (ms)": speed_inference,
            "Fitness": fitness
        })

    # 3. COMPARAISON ET AFFICHAGE
    # Conversion des résultats en DataFrame Pandas pour analyse analytique
    df_compare = pd.DataFrame(results_summary)
    print("\n" + "="*50)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("="*50)
    print(df_compare.to_string(index=False))

    # Sauvegarde du rapport en CSV
    os.makedirs("comparaison_models", exist_ok=True)
    df_compare.to_csv("comparaison_models/rapport_comparatif.csv", index=False)

    # 4. GÉNÉRATION DE GRAPHIQUES DE COMPARAISON
    plt.figure(figsize=(10, 5))

    # Graphique : mAP50 vs Vitesse d'inférence
    plt.subplot(1, 2, 1)
    plt.bar(df_compare["Modèle"], df_compare["mAP50"])
    plt.title("Précision Globale (mAP50)")
    plt.ylabel("Score (0 à 1)")

    plt.subplot(1, 2, 2)
    plt.bar(df_compare["Modèle"], df_compare["Vitesse Inférence (ms)"])
    plt.title("Temps d'inférence (Plus bas = Plus rapide)")
    plt.ylabel("Millisecondes (ms)")

    plt.tight_layout()
    plt.savefig("comparaison_models/graphique_comparatif.png")
    plt.show()

    print("\nAnalyse terminée ! Les rapports et graphiques ont été sauvegardés dans le dossier 'comparaison_models/'.")