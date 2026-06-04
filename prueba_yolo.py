from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar modelo
modelo = YOLO('./runs/best.pt')
img = cv2.imread('R.jpg')

print("🔍 Analizando imagen con diferentes niveles de confianza:\n")

# Probar diferentes umbrales de confianza
for conf in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    resultados = modelo.predict(img, conf=conf, verbose=False)
    num_detections = len(resultados[0].boxes)

    if num_detections > 0:
        print(f"✅ Confianza {conf:.2f}: {num_detections} detección(es)")
        for box in resultados[0].boxes:
            print(f"   - Clase: {box.cls.item()}, Confianza: {box.conf.item():.3f}")
    else:
        print(f"❌ Confianza {conf:.2f}: Sin detecciones")

# Guardar imagen con detecciones (confianza muy baja)
print("\n📸 Generando imagen con detecciones (conf=0.01)...")
resultados = modelo.predict(img, conf=0.01, save=True, project='.', name='analisis_yolo')
print("✅ Imagen guardada en: analisis_yolo/R.jpg")