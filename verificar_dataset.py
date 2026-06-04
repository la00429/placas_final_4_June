import os
from pathlib import Path
import yaml

print("=" * 60)
print("🔍 VERIFICACIÓN DEL DATASET")
print("=" * 60)

# Verificar carpetas
carpetas = {
    'train/images': 0,
    'train/labels': 0,
    'valid/images': 0,
    'valid/labels': 0,
    'test/images': 0,
    'test/labels': 0
}

for carpeta, count in carpetas.items():
    ruta = Path(f"./{carpeta}")
    if ruta.exists():
        archivos = list(ruta.glob("*.*"))
        print(f"✅ {carpeta}/ : {len(archivos)} archivos")
    else:
        print(f"❌ Falta: {carpeta}/")

# Verificar data.yaml
yaml_path = Path("./data.yaml")
if yaml_path.exists():
    print(f"\n✅ data.yaml existe")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"\n📄 Configuración:")
        print(f"   train: {config.get('train')}")
        print(f"   val: {config.get('val')}")
        print(f"   test: {config.get('test')}")
        print(f"   Clases (nc): {config.get('nc')}")
        print(f"   Nombres: {config.get('names')}")

    except Exception as e:
        print(f"\n❌ Error al leer data.yaml: {e}")
else:
    print(f"\n❌ Falta data.yaml")

print("\n" + "=" * 60)
print("✅ Verificación completada")
print("=" * 60)