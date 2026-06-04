# verificar_clases.py
import os
from pathlib import Path


def verificar_clases():
    """Verifica las clases en los labels."""

    print("🔍 Verificando clases en los labels...\n")

    clases_encontradas = set()

    for split in ['train', 'valid', 'test']:
        labels_dir = Path(f"./{split}/labels")
        if labels_dir.exists():
            txt_files = list(labels_dir.glob("*.txt"))
            for txt_file in txt_files[:5]:  # Revisar primeros 5 archivos
                with open(txt_file, 'r') as f:
                    lineas = f.readlines()
                    for linea in lineas:
                        if linea.strip():
                            clase_id = int(linea.split()[0])
                            clases_encontradas.add(clase_id)

    print(f"Clases encontradas en labels: {sorted(clases_encontradas)}")
    print(f"Número total de clases: {len(clases_encontradas)}")

    if len(clases_encontradas) == 1:
        print("\n✅ Tu dataset tiene 1 clase: 'placa'")
        print("💡 Usa la OPCIÓN 1 en data.yaml")
    elif len(clases_encontradas) == 2:
        print("\n✅ Tu dataset tiene 2 clases")
        print("💡 Usa la OPCIÓN 2 en data.yaml")
        print("💡 Asegúrate de que los nombres coincidan:")
        print("   0: placa_carro")
        print("   1: placa_moto")
    else:
        print(f"\n⚠️  Tu dataset tiene {len(clases_encontradas)} clases")


if __name__ == "__main__":
    verificar_clases()