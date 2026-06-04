"""
Aplicación Streamlit para detección de placas colombianas
Versión: Producción con OCR mejorado
"""
import streamlit as st
from datetime import datetime
import re
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Detector Placas Colombia", page_icon="🇨🇴", layout="wide")


# ==========================================
# CARGAR MODELOS
# ==========================================
@st.cache_resource
def cargar_modelos():
    """Carga YOLO y OCR en caché."""
    with st.spinner("Cargando modelos de IA..."):
        # OCR
        ocr_reader = easyocr.Reader(['es'], gpu=False)

        # YOLO
        yolo_detector = None
        modelo_path = 'runs/best.pt'

        if os.path.exists(modelo_path):
            try:
                yolo_detector = YOLO(modelo_path)
                st.sidebar.success("✅ Modelo YOLO cargado")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error: {e}")
        else:
            st.sidebar.warning("⚠️ Sin modelo YOLO")

        return ocr_reader, yolo_detector


reader, detector = cargar_modelos()


# ==========================================
# FUNCIONES
# ==========================================
def clasificar_por_formato(placa):
    """Clasifica carro vs moto."""
    placa_limpia = re.sub(r'[^A-Z0-9]', '', placa.upper())

    if re.match(r"^[A-Z]{3}\d{2}[A-Z]$", placa_limpia):
        return "🏍️ Motocicleta", "Moto"
    elif re.match(r"^[A-Z]{3}\d{3}[A-Z]$", placa_limpia):
        return "🚗 Vehículo Particular", "Carro (Mercosur)"
    elif re.match(r"^[A-Z]{3}\d{3}$", placa_limpia):
        return "🚗 Vehículo Particular", "Carro (Antiguo)"
    elif re.match(r"^[A-Z]{2}\d{3}[A-Z]$", placa_limpia):
        return "🚕 Servicio Público", "Público"
    else:
        return "❓ Tipo desconocido", "Desconocido"


def analizar_pico_y_placa(tipo_vehiculo, placa):
    """Calcula Pico y Placa."""
    placa_limpia = re.sub(r'[^A-Z0-9]', '', placa.upper())

    if tipo_vehiculo == "🏍️ Motocicleta":
        ultimo_numero = int(placa_limpia[-2]) if len(placa_limpia) >= 2 else 0
    else:
        numeros = re.findall(r'\d', placa_limpia)
        ultimo_numero = int(numeros[-1]) if numeros else 0

    dia_semana = datetime.now().weekday()
    hora_actual = datetime.now().hour

    tabla_carros = {
        0: [4, 5, 6, 7],
        1: [8, 9, 0, 1],
        2: [2, 3, 4, 5],
        3: [6, 7, 8, 9],
        4: [0, 1, 2, 3],
    }

    if dia_semana >= 5:
        if tipo_vehiculo == "🏍️ Motocicleta":
            if ultimo_numero in [0, 1, 2, 3, 4]:
                return f"🚫 RESTRICCIÓN (Sábado)", ultimo_numero
            else:
                return f"✅ Libre (Sábado)", ultimo_numero
        return f"✅ Libre (Fin de semana)", ultimo_numero

    if not (6 <= hora_actual < 20):
        return f"✅ Libre (Fuera de horario)", ultimo_numero

    if tipo_vehiculo == "🏍️ Motocicleta":
        if dia_semana < 5:
            return f"✅ Libre (Motos solo sábados)", ultimo_numero
    else:
        if ultimo_numero in tabla_carros[dia_semana]:
            return f"🚫 RESTRICCIÓN ACTIVA", ultimo_numero
        else:
            return f"✅ Libre Circulación", ultimo_numero

    return f"✅ Libre", ultimo_numero


def preprocess_image_avanzado(img):
    """Preprocesamiento avanzado para OCR."""
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Desenfocado bilateral (reduce ruido preservando bordes)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Umbral adaptativo
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Volver a BGR para compatibilidad
    img_procesada = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return img_procesada


# ==========================================
# INTERFAZ
# ==========================================
st.title("🇨🇴 Detector de Placas Colombianas")
st.markdown("**Detección con YOLO + OCR Mejorado**")

with st.sidebar:
    st.header("📋 Pico y Placa Bogotá")
    st.markdown("""
    **🚗 Carros (Lun-Vie 6am-8pm):**
    - Lun: 4-5-6-7
    - Mar: 8-9-0-1
    - Mié: 2-3-4-5
    - Jue: 6-7-8-9
    - Vie: 0-1-2-3

    **🏍️ Motos (Sábados):**
    - 0-1-2-3-4: Restringidas
    - 5-6-7-8-9: Libres
    """)

    st.divider()

    confianza = st.slider("Confianza YOLO", 0.05, 0.9, 0.25, 0.05)
    mostrar_preprocesada = st.checkbox("Mostrar imagen preprocesada", False)

archivo = st.file_uploader("📷 Sube imagen", type=["jpg", "jpeg", "png"])

if archivo:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(archivo, caption="Imagen original", width='stretch')

        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, 1)
        img_procesada = preprocess_image_avanzado(img_original)

        if mostrar_preprocesada:
            st.image(img_procesada, caption="Imagen preprocesada", width='stretch')

    with st.spinner("🔍 Procesando..."):
        detecciones = []
        metodo = ""

        # INTENTO 1: YOLO
        if detector is not None:
            resultados = detector(img_original, conf=confianza, verbose=False)

            for r in resultados:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_yolo = float(box.conf[0])

                    placa_crop = img_original[y1:y2, x1:x2]
                    placa_procesada = preprocess_image_avanzado(placa_crop)

                    textos = reader.readtext(placa_procesada)

                    for (bbox, texto, conf_ocr) in textos:
                        tipo, formato = clasificar_por_formato(texto)
                        if tipo != "❓ Tipo desconocido" and conf_ocr > 0.3:
                            estado, digito = analizar_pico_y_placa(tipo, texto)
                            detecciones.append({
                                "placa": re.sub(r'[^A-Z0-9]', '', texto.upper()),
                                "tipo": tipo,
                                "formato": formato,
                                "digito": digito,
                                "estado": estado,
                                "conf_yolo": conf_yolo,
                                "conf_ocr": conf_ocr
                            })
                            metodo = "YOLO + OCR"

        # INTENTO 2: OCR DIRECTO
        if not detecciones:
            st.info("💡 YOLO no detectó. Usando OCR directo...")

            img_ocr = preprocess_image_avanzado(img_original)
            textos = reader.readtext(img_ocr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            for (bbox, texto, conf_ocr) in textos:
                tipo, formato = clasificar_por_formato(texto)

                if tipo != "❓ Tipo desconocido" and conf_ocr > 0.4:
                    estado, digito = analizar_pico_y_placa(tipo, texto)
                    detecciones.append({
                        "placa": re.sub(r'[^A-Z0-9]', '', texto.upper()),
                        "tipo": tipo,
                        "formato": formato,
                        "digito": digito,
                        "estado": estado,
                        "conf_yolo": 0.0,
                        "conf_ocr": conf_ocr
                    })
                    metodo = "OCR Directo"

        with col2:
            st.subheader("📊 Resultados")

            if detecciones:
                st.success(f"✅ Método: {metodo}")

                for i, det in enumerate(detecciones, 1):
                    with st.expander(f"🔢 Placa {i}: {det['placa']}", expanded=True):
                        st.success(f"**Placa:** {det['placa']}")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Tipo", det["tipo"])
                            st.metric("Formato", det["formato"])
                        with col_b:
                            st.metric("Dígito", det["digito"])

                        st.divider()

                        if "RESTRICCIÓN" in det['estado']:
                            st.error(f"⚠️ **{det['estado']}**")
                        else:
                            st.success(f"✅ **{det['estado']}**")

                        st.caption(f"""
                        🔍 YOLO: {det['conf_yolo']:.2%} | 
                        📝 OCR: {det['conf_ocr']:.2%}
                        """)
            else:
                st.error("❌ No se detectó placa")
                st.warning("""
                💡 **Posibles causas:**
                - Placa muy desgastada o sucia
                - Mala iluminación
                - Ángulo muy inclinado
                - Resolución muy baja

                **Soluciones:**
                - Toma la foto más de cerca
                - Mejor iluminación
                - Ángulo frontal
                """)

st.divider()
st.caption("YOLOv8 + EasyOCR | RUNT Colombia")