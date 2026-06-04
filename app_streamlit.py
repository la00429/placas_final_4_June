"""
Aplicación Streamlit para detección de placas colombianas
YOLOv8 + EasyOCR
"""

import streamlit as st
from datetime import datetime
import re
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# CONFIGURACIÓN STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Detector Placas Colombia",
    page_icon="🇨🇴",
    layout="wide"
)

# ==========================================
# CARGAR MODELOS
# ==========================================

@st.cache_resource
def cargar_modelos():

    with st.spinner("Cargando modelos IA..."):

        # OCR
        reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        # ==================================
        # CARGAR MEJOR MODELO YOLO
        # ==================================

        modelo_path = None

        # prioridad 1
        if Path("runs/best.pt").exists():
            modelo_path = "runs/best.pt"

        # prioridad 2
        elif Path("modelo_final/best.pt").exists():
            modelo_path = "modelo_final/best.pt"

        # prioridad 3
        else:
            runs_dir = Path("runs/detect")

            experimentos = list(
                runs_dir.glob("*/weights/best.pt")
            )

            if experimentos:
                modelo_path = str(
                    max(
                        experimentos,
                        key=lambda p: p.stat().st_mtime
                    )
                )

        # cargar modelo
        if modelo_path:
            detector = YOLO(modelo_path)
            st.sidebar.success(f"✅ Modelo cargado")
            st.sidebar.code(modelo_path)

        else:
            detector = YOLO("yolov8s.pt")
            st.sidebar.warning(
                "⚠️ Usando YOLO genérico"
            )

        return reader, detector


reader, detector = cargar_modelos()

# ==========================================
# ANALIZAR PLACA
# ==========================================

def analizar_placa(texto_ocr):

    placa = re.sub(
        r'[^A-Z0-9]',
        '',
        texto_ocr.upper()
    )

    # ======================================
    # FORMATO PLACAS COLOMBIA
    # ======================================

    es_carro_antiguo = bool(
        re.match(r"^[A-Z]{3}\d{3}$", placa)
    )

    es_carro_nuevo = bool(
        re.match(r"^[A-Z]{3}\d{3}[A-Z]$", placa)
    )

    es_moto = bool(
        re.match(r"^[A-Z]{3}\d{2}[A-Z]$", placa)
    )

    if not (
        es_carro_antiguo
        or es_carro_nuevo
        or es_moto
    ):
        return None

    # ======================================
    # DETERMINAR TIPO
    # ======================================

    if es_moto:
        tipo = "Motocicleta"
        digito = int(placa[-2])
        formato = "Moto"

    elif es_carro_nuevo:
        tipo = "Vehículo Particular"
        digito = int(placa[-2])
        formato = "Mercosur"

    else:
        tipo = "Vehículo Particular"
        digito = int(placa[-1])
        formato = "Antiguo"

    # ======================================
    # PICO Y PLACA
    # ======================================

    dia_semana = datetime.now().weekday()
    hora_actual = datetime.now().hour

    en_horario = 6 <= hora_actual < 20

    tabla_restriccion = {
        0: [4, 5, 6, 7],
        1: [8, 9, 0, 1],
        2: [2, 3, 4, 5],
        3: [6, 7, 8, 9],
        4: [0, 1, 2, 3],
    }

    if dia_semana >= 5:
        estado = "✅ Libre (Fin de semana)"

    elif not en_horario:
        estado = "✅ Libre (Fuera de horario)"

    elif digito in tabla_restriccion[dia_semana]:
        estado = "🚫 RESTRICCIÓN ACTIVA"

    else:
        estado = "✅ Libre Circulación"

    return {
        "placa": placa,
        "tipo": tipo,
        "digito": digito,
        "estado": estado,
        "formato": formato
    }

# ==========================================
# INTERFAZ
# ==========================================

st.title("🇨🇴 Detector de Placas Colombianas")

st.markdown(
    """
Sistema automático de:

- detección de placas
- OCR
- validación formato
- pico y placa
"""
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Configuración")

    confianza = st.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05
    )

    mostrar_recortes = st.checkbox(
        "Mostrar recortes OCR",
        value=True
    )

# ==========================================
# SUBIR IMAGEN
# ==========================================

archivo = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PROCESAMIENTO
# ==========================================

if archivo:

    col1, col2 = st.columns(2)

    # ======================================
    # LEER IMAGEN
    # ======================================

    file_bytes = np.asarray(
        bytearray(archivo.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    img_original = img.copy()

    # ======================================
    # DETECCIÓN YOLO
    # ======================================

    with st.spinner("Detectando placas..."):

        resultados = detector(
            img,
            conf=confianza,
            imgsz=960
        )

    detecciones = []

    # ======================================
    # RECORRER DETECCIONES
    # ======================================

    for r in resultados:

        for box in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf_det = float(box.conf[0])

            # ==================================
            # DIBUJAR BOUNDING BOX
            # ==================================

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================
            # RECORTE PLACA
            # ==================================

            placa_crop = img_original[
                y1:y2,
                x1:x2
            ]

            if placa_crop.size == 0:
                continue

            # ==================================
            # PREPROCESAMIENTO OCR
            # ==================================

            gray = cv2.cvtColor(
                placa_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            gray = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # ==================================
            # OCR
            # ==================================

            textos = reader.readtext(gray)

            if mostrar_recortes:
                st.image(
                    gray,
                    caption="Recorte OCR",
                    width=300
                )

            # ==================================
            # ANALIZAR OCR
            # ==================================

            for (_, texto, conf_ocr) in textos:

                if conf_ocr < 0.30:
                    continue

                analisis = analizar_placa(texto)

                if analisis:

                    detecciones.append({

                        "placa": analisis["placa"],
                        "tipo": analisis["tipo"],
                        "digito": analisis["digito"],
                        "estado": analisis["estado"],
                        "formato": analisis["formato"],
                        "conf_det": conf_det,
                        "conf_ocr": conf_ocr
                    })

    # ======================================
    # MOSTRAR RESULTADOS
    # ======================================

    with col1:

        st.subheader("📷 Imagen procesada")

        st.image(
            img,
            channels="BGR",
            use_column_width=True
        )

    with col2:

        st.subheader("📋 Resultados")

        if detecciones:

            for i, det in enumerate(detecciones, 1):

                with st.expander(
                    f"Placa {i}: {det['placa']}",
                    expanded=True
                ):

                    st.success(
                        f"Placa detectada: {det['placa']}"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Tipo",
                        det["tipo"]
                    )

                    c2.metric(
                        "Formato",
                        det["formato"]
                    )

                    c3.metric(
                        "Dígito",
                        det["digito"]
                    )

                    st.divider()

                    if "RESTRICCIÓN" in det["estado"]:
                        st.error(det["estado"])
                    else:
                        st.success(det["estado"])

                    st.caption(
                        f"""
Detección: {det['conf_det']:.2%}
OCR: {det['conf_ocr']:.2%}
"""
                    )

        else:

            st.error(
                "❌ No se detectó ninguna placa válida"
            )

            st.info(
                """
Sugerencias:

- buena iluminación
- placa visible
- evitar imágenes borrosas
- evitar ángulos extremos
"""
            )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "YOLOv8 + EasyOCR | Detección Placas Colombia"
)
