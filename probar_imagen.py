"""
Script para probar el modelo con imágenes individuales
"""
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def probar_imagen(ruta_imagen, modelo_path=None):
    """Prueba el modelo con una imagen específica."""

    # Buscar modelo
    if modelo_path is None:
        runs_dir = Path('runs/detect')
        experimentos = list(runs_dir.glob('*/weights/best.pt'))
        if experimentos:
            modelo_path = str(max(experimentos, key=lambda p: p.stat().st_mtime))
        else:
            print("❌ No se encontró ningún modelo entrenado")
            return

    print(f"📥 Cargando modelo: {modelo_path}")
    modelo = YOLO(modelo_path)

    # Verificar imagen
    if not Path(ruta_imagen).exists():
        print(f"❌ No se encontró la imagen: {ruta_imagen}")
        return

    print(f"\n🖼️ Procesando imagen: {ruta_imagen}")

    # Predecir
    resultados = modelo.predict(
        source=ruta_imagen,
        conf=0.5,  # Confianza mínima 50%
        iou=0.45,  # IoU threshold
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        save=True,
        project='runs/predict',
        name='prueba_individual',
        exist_ok=True
    )

    # Mostrar resultados
    print(f"\n✅ Predicción completada!")
    print(f"📁 Resultado guardado en: runs/predict/prueba_individual/")

    for r in resultados:
        cajas = r.boxes
        if len(cajas) == 0:
            print("\n⚠️  No se detectó ninguna placa")
        else:
            print(f"\n🎯 Detecciones encontradas: {len(cajas)}")
            for i, box in enumerate(cajas, 1):
                clase = int(box.cls[0])
                conf = float(box.conf[0])
                nombre_clase = r.names[clase]
                print(f"   {i}. {nombre_clase}: {conf:.2%} de confianza")

    # Mostrar imagen con predicciones
    print("\n Mostrando imagen con predicciones...")
    img = cv2.imread(ruta_imagen)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title("Predicción del Modelo")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        probar_imagen(sys.argv[1])
    else:
        print("Uso: python probar_imagen.py <ruta_imagen>")
        print("\nEjemplo: python probar_imagen.py test/images/imagen1.jpg")