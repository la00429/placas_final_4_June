"""
Script para probar el modelo directamente
"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar modelo
modelo = YOLO('runs/best.pt')
print("✅ Modelo cargado correctamente")

# Cargar imagen
img = cv2.imread('R.jpg')  # Asegúrate de que R.jpg esté en la misma carpeta

if img is None:
    print("❌ No se encontró la imagen R.jpg")
    print("💡 Copia la imagen R.jpg a la carpeta del proyecto")
else:
    print(f"📷 Imagen cargada: {img.shape}")

    # Predecir
    resultados = modelo.predict(img, conf=0.25, save=True)

    # Mostrar resultados
    for r in resultados:
        print(f"\n🎯 Detecciones: {len(r.boxes)}")
        for box in r.boxes:
            clase = int(box.cls[0])
            conf = float(box.conf[0])
            nombre = r.names[clase]
            print(f"   - {nombre}: {conf:.2%}")

    # Mostrar imagen
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecciones: {len(r.boxes)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()