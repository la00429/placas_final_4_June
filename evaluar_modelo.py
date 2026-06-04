"""
Script para evaluar el modelo entrenado
"""
from ultralytics import YOLO
import torch


def evaluar_modelo():
    """Evalúa el modelo en el dataset de test."""

    print("=" * 60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 60)

    # Buscar el mejor modelo
    modelo_path = 'runs/detect/detector_placas/weights/best.pt'

    # Si no existe, buscar el más reciente
    from pathlib import Path
    runs_dir = Path('runs/detect')
    if not Path(modelo_path).exists():
        experimentos = list(runs_dir.glob('*/weights/best.pt'))
        if experimentos:
            modelo_path = str(max(experimentos, key=lambda p: p.stat().st_mtime))
            print(f" Usando modelo más reciente: {modelo_path}")
        else:
            print("❌ No se encontró ningún modelo entrenado")
            print("💡 Ejecuta main.py primero para entrenar el modelo")
            return

    # Cargar modelo
    print(f"\n📥 Cargando modelo: {modelo_path}")
    modelo = YOLO(modelo_path)

    # Evaluar
    print("\n Evaluando en dataset de test...")
    metricas = modelo.val(
        data='data.yaml',
        split='test',
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_json=True,
        save_hybrid=True
    )

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📈 MÉTRICAS DE EVALUACIÓN:")
    print("=" * 60)
    print(f"   mAP50-95: {metricas.box.map:.4f}")
    print(f"   mAP50:    {metricas.box.map50:.4f}")
    print(f"   mAP75:    {metricas.box.map75:.4f}")
    print(f"   Precision: {metricas.box.mp:.4f}")
    print(f"   Recall:    {metricas.box.mr:.4f}")
    print("=" * 60)

    # Interpretación
    print("\n💡 INTERPRETACIÓN:")
    if metricas.box.map50 > 0.9:
        print("   ✅ Excelente modelo (mAP50 > 0.9)")
    elif metricas.box.map50 > 0.7:
        print("   ✅ Buen modelo (mAP50 > 0.7)")
    elif metricas.box.map50 > 0.5:
        print("   ⚠️  Modelo aceptable (mAP50 > 0.5)")
    else:
        print("   ⚠️  Modelo necesita más entrenamiento")

    print(f"\n Resultados guardados en: {Path(modelo_path).parent.parent}")


if __name__ == "__main__":
    evaluar_modelo()