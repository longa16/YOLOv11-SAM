from ultralytics import YOLO
from pathlib import Path

def main():
    model_path = Path(__file__).parent / 'yolo11s-seg.pt'
    model = YOLO(str(model_path))
    

    results = model.train(
        data="C:\\Users\\loic.ngassa\\Downloads\\Cailloux.v1i.yolov11\\data.yaml", 
        epochs=100, 
        imgsz=640, 
        workers=2
    )

if __name__ == '__main__':
    main()

print("Training completed successfully.")