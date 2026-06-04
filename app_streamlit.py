"""
APP.PY
Detector Inteligente de Placas Colombianas
YOLOv8 + EasyOCR + Clasificación Colombiana
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
        # BUSCAR MODELO ENTRENADO
        # ==================================

        modelo_path = None

        if Path("runs/best.pt").exists():

            modelo_path = "runs/best.pt"

        elif Path("modelo_final/best.pt").exists():

            modelo_path = "modelo_final/best.pt"

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

        # ==================================
        # CARGAR YOLO
        # ==================================

        if modelo_path:

            detector = YOLO(modelo_path)

            st.sidebar.success(
                "✅ Modelo entrenado cargado"
            )

            st.sidebar.code(modelo_path)

        else:

            detector = YOLO("yolov8s.pt")

            st.sidebar.warning(
                "⚠️ Usando YOLO genérico"
            )

        return reader, detector


reader, detector = cargar_modelos()

# ==========================================
# CLASIFICADOR COLOMBIANO
# ==========================================

def clasificar_tipo_placa(placa_crop, texto):

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    hsv = cv2.cvtColor(
        placa_crop,
        cv2.COLOR_BGR2HSV
    )

    # ======================================
    # AMARILLO
    # ======================================

    amarillo_bajo = np.array([15, 80, 80])
    amarillo_alto = np.array([40, 255, 255])

    mask_amarillo = cv2.inRange(
        hsv,
        amarillo_bajo,
        amarillo_alto
    )

    amarillo = (
        np.sum(mask_amarillo > 0)
        / mask_amarillo.size
    )

    # ======================================
    # BLANCO
    # ======================================

    blanco_bajo = np.array([0, 0, 170])
    blanco_alto = np.array([180, 50, 255])

    mask_blanco = cv2.inRange(
        hsv,
        blanco_bajo,
        blanco_alto
    )

    blanco = (
        np.sum(mask_blanco > 0)
        / mask_blanco.size
    )

    # ======================================
    # AZUL
    # ======================================

    azul_bajo = np.array([90, 50, 50])
    azul_alto = np.array([140, 255, 255])

    mask_azul = cv2.inRange(
        hsv,
        azul_bajo,
        azul_alto
    )

    azul = (
        np.sum(mask_azul > 0)
        / mask_azul.size
    )

    # ======================================
    # VERDE
    # ======================================

    verde_bajo = np.array([35, 40, 40])
    verde_alto = np.array([90, 255, 255])

    mask_verde = cv2.inRange(
        hsv,
        verde_bajo,
        verde_alto
    )

    verde = (
        np.sum(mask_verde > 0)
        / mask_verde.size
    )

    # ======================================
    # ROJO
    # ======================================

    rojo1_bajo = np.array([0, 70, 50])
    rojo1_alto = np.array([10, 255, 255])

    rojo2_bajo = np.array([170, 70, 50])
    rojo2_alto = np.array([180, 255, 255])

    mask_rojo1 = cv2.inRange(
        hsv,
        rojo1_bajo,
        rojo1_alto
    )

    mask_rojo2 = cv2.inRange(
        hsv,
        rojo2_bajo,
        rojo2_alto
    )

    mask_rojo = mask_rojo1 + mask_rojo2

    rojo = (
        np.sum(mask_rojo > 0)
        / mask_rojo.size
    )

    # ======================================
    # DEBUG VISUAL
    # ======================================

    st.write("🟡 Amarillo:", round(amarillo, 3))
    st.write("⚪ Blanco:", round(blanco, 3))
    st.write("🔵 Azul:", round(azul, 3))
    st.write("🟢 Verde:", round(verde, 3))
    st.write("🔴 Rojo:", round(rojo, 3))

    # ======================================
    # MOTO
    # ======================================

    es_moto = bool(
        re.match(
            r"^[A-Z]{3}\d{2}[A-Z]$",
            texto
        )
    )

    if es_moto:

        if amarillo > 0.15:
            return "Motocicleta Particular"

        elif blanco > 0.15:
            return "Motocicleta Pública"

        else:
            return "Motocicleta"

    # ======================================
    # DIPLOMÁTICAS
    # ======================================

    if azul > 0.20:

        if texto.startswith("CD"):
            return "Cuerpo Diplomático"

        elif texto.startswith("CC"):
            return "Cuerpo Consular"

        elif texto.startswith("OI"):
            return "Organismo Internacional"

        elif texto.startswith("AT"):
            return "Personal Administrativo Diplomático"

        else:
            return "Vehículo Diplomático"

    # ======================================
    # REMOLQUES
    # ======================================

    if verde > 0.20:

        if texto.startswith("R"):
            return "Remolque"

        elif texto.startswith("S"):
            return "Semirremolque"

        else:
            return "Vehículo Oficial"

    # ======================================
    # CARGA PÚBLICA
    # ======================================

    if rojo > 0.20:
        return "Vehículo de Carga Pública"

    # ======================================
    # SERVICIO PÚBLICO
    # ======================================

    if blanco > 0.25:
        return "Servicio Público"

    # ======================================
    # PARTICULAR
    # ======================================

    if amarillo > 0.15:
        return "Vehículo Particular"

    # ======================================
    # CLÁSICOS
    # ======================================

    if azul > 0.08 and blanco > 0.08:
        return "Vehículo Clásico"

    return "Desconocido"

# ==========================================
# ANALIZAR PLACA
# ==========================================

def analizar_placa(texto_ocr):

    placa = texto_ocr.strip().upper()

    placa = re.sub(
        r'[^A-Z0-9]',
        '',
        placa
    )

    st.write("🔎 OCR limpio:", placa)

    # ======================================
    # FORMATO AAA999
    # ======================================

    if len(placa) == 6:

        letras = placa[:3]
        numeros = placa[3:]

        letras = letras.replace("0", "O")
        letras = letras.replace("1", "I")
        letras = letras.replace("2", "Z")
        letras = letras.replace("8", "B")

        numeros = numeros.replace("O", "0")
        numeros = numeros.replace("I", "1")
        numeros = numeros.replace("Z", "2")
        numeros = numeros.replace("B", "8")

        placa = letras + numeros

    # ======================================
    # FORMATO AAA999A
    # ======================================

    elif len(placa) == 7:

        letras = placa[:3]
        numeros = placa[3:6]
        final = placa[6]

        letras = letras.replace("0", "O")
        letras = letras.replace("1", "I")

        numeros = numeros.replace("O", "0")
        numeros = numeros.replace("I", "1")

        final = final.replace("0", "O")
        final = final.replace("1", "I")

        placa = letras + numeros + final

    st.success(f"✅ OCR corregido: {placa}")

    # ======================================
    # VALIDAR FORMATOS
    # ======================================

    es_carro = bool(
        re.match(
            r"^[A-Z]{3}\d{3}$",
            placa
        )
    )

    es_mercosur = bool(
        re.match(
            r"^[A-Z]{3}\d{3}[A-Z]$",
            placa
        )
    )

    es_moto = bool(
        re.match(
            r"^[A-Z]{3}\d{2}[A-Z]$",
            placa
        )
    )

    if not (
        es_carro
        or es_mercosur
        or es_moto
    ):

        st.error(
            f"❌ Formato inválido: {placa}"
        )

        return None

    # ======================================
    # DÍGITO RESTRICCIÓN
    # ======================================

    if es_moto:

        digito = int(placa[-2])

        formato = "Moto"

    elif es_mercosur:

        digito = int(placa[-2])

        formato = "Mercosur"

    else:

        digito = int(placa[-1])

        formato = "Tradicional"

    # ======================================
    # PICO Y PLACA
    # ======================================

    dia_semana = datetime.now().weekday()

    hora_actual = datetime.now().hour

    en_horario = 6 <= hora_actual < 20

    tabla = {

        0: [4, 5, 6, 7],
        1: [8, 9, 0, 1],
        2: [2, 3, 4, 5],
        3: [6, 7, 8, 9],
        4: [0, 1, 2, 3]
    }

    if dia_semana >= 5:

        estado = "✅ Libre (fin de semana)"

    elif not en_horario:

        estado = "✅ Libre (fuera horario)"

    elif digito in tabla[dia_semana]:

        estado = "🚫 RESTRICCIÓN ACTIVA"

    else:

        estado = "✅ Libre circulación"

    return {

        "placa": placa,
        "digito": digito,
        "estado": estado,
        "formato": formato
    }

# ==========================================
# INTERFAZ
# ==========================================

st.title("🇨🇴 Detector Inteligente de Placas")

st.markdown("""
Sistema avanzado de:

- detección automática
- OCR robusto
- clasificación colombiana
- validación inteligente
- pico y placa
""")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Configuración")

    confianza = st.slider(
        "Confianza mínima",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )

    mostrar_ocr = st.checkbox(
        "Mostrar OCR",
        value=True
    )

# ==========================================
# UPLOAD
# ==========================================

archivo = st.file_uploader(
    "📤 Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PROCESAR
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
    # DETECCIÓN
    # ======================================

    with st.spinner("🔍 Detectando placas..."):

        resultados = detector(
            img,
            conf=confianza,
            imgsz=1280
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

            conf_det = float(
                box.conf[0]
            )

            # ==================================
            # DIBUJAR
            # ==================================

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================
            # RECORTE
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
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            imagenes_ocr = []

            # normal
            imagenes_ocr.append(gray)

            # threshold
            thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            imagenes_ocr.append(thresh)

            # invertida
            invertida = cv2.bitwise_not(
                thresh
            )

            imagenes_ocr.append(
                invertida
            )

            # adaptive
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            imagenes_ocr.append(
                adaptive
            )

            # sharpen
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])

            sharp = cv2.filter2D(
                gray,
                -1,
                kernel
            )

            imagenes_ocr.append(
                sharp
            )

            # ==================================
            # MOSTRAR OCR
            # ==================================

            if mostrar_ocr:

                st.image(
                    thresh,
                    caption="OCR",
                    width=300
                )

            # ==================================
            # OCR MÚLTIPLE
            # ==================================

            mejor_texto = ""
            mejor_conf = 0

            for img_ocr in imagenes_ocr:

                resultados_ocr = reader.readtext(
                    img_ocr,
                    detail=1
                )

                for (_, texto, conf_ocr) in resultados_ocr:

                    texto = re.sub(
                        r'[^A-Z0-9]',
                        '',
                        texto.upper()
                    )

                    st.write(
                        "🧠 OCR:",
                        texto
                    )

                    if len(texto) < 5:
                        continue

                    if conf_ocr > mejor_conf:

                        mejor_conf = conf_ocr
                        mejor_texto = texto

            # ==================================
            # MEJOR OCR
            # ==================================

            if mejor_texto:

                st.success(
                    f"✅ Mejor OCR: {mejor_texto}"
                )

                analisis = analizar_placa(
                    mejor_texto
                )

                if analisis:

                    tipo_vehiculo = clasificar_tipo_placa(
                        placa_crop,
                        mejor_texto
                    )

                    detecciones.append({

                        "placa": analisis["placa"],
                        "tipo": tipo_vehiculo,
                        "digito": analisis["digito"],
                        "estado": analisis["estado"],
                        "formato": analisis["formato"],
                        "conf_det": conf_det,
                        "conf_ocr": mejor_conf
                    })

    # ======================================
    # MOSTRAR IMAGEN
    # ======================================

    with col1:

        st.subheader(
            "📷 Imagen procesada"
        )

        st.image(
            img,
            channels="BGR",
            use_container_width=True
        )

    # ======================================
    # RESULTADOS
    # ======================================

    with col2:

        st.subheader("📋 Resultados")

        if detecciones:

            for i, det in enumerate(
                detecciones,
                1
            ):

                with st.expander(
                    f"Placa {i}: {det['placa']}",
                    expanded=True
                ):

                    st.success(
                        f"✅ {det['placa']}"
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

                        st.error(
                            det["estado"]
                        )

                    else:

                        st.success(
                            det["estado"]
                        )

                    st.caption(
                        f"""
Detección:
{det['conf_det']:.2%}

OCR:
{det['conf_ocr']:.2%}
"""
                    )

        else:

            st.error(
                "❌ No se detectó ninguna placa válida"
            )

            st.info("""
💡 Recomendaciones:

- usar fotos más cercanas
- evitar blur
- buena iluminación
- evitar inclinación
""")

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "YOLOv8 + EasyOCR + Clasificación Colombiana"
)