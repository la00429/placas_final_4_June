"""
Script de diagnóstico con preprocessing de imagen
"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(img):
    """Mejora la imagen para facilitar la detección."""
    # Convertir a HSV para mejorar contraste
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Aumentar saturación y brillo
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Más saturación
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)  # Más brillo

    # Volver a BGR
    img_mejorada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Aplicar sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_mejorada, -1, kernel)

    return img_sharp


# Cargar modelo
modelo = YOLO('runs/best.pt')
print("✅ Modelo cargado")

# Cargar imagen
img = cv2.imread('R.jpg')
print(f"📷 Imagen original: {img.shape}")

# Preprocesar
img_procesada = preprocess_image(img)
print("✨ Imagen mejorada")

# Probar con diferentes niveles de confianza
confi_levels = [0.05, 0.1, 0.15, 0.2, 0.25]

for conf in confi_levels:
    resultados = modelo.predict(img_procesada, conf=conf, verbose=False)

    for r in resultados:
        num_detections = len(r.boxes)
        print(f"Confianza {conf:.2f}: {num_detections} detecciones")

        if num_detections > 0:
            print(f"  ✅ ¡DETECTADO!")
            for box in r.boxes:
                clase = int(box.cls[0])
                confianza = float(box.conf[0])
                nombre = r.names[clase]
                print(f"     - {nombre}: {confianza:.2%}")

            # Mostrar imagen con detecciones
            img_annotated = r.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
            plt.title(f"Detecciones con confianza {conf}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            break  # Salir del loop si detectó algo

print("\n💡 Si no detecta con ninguna confianza, el modelo necesita más datos de placas amarillas.")