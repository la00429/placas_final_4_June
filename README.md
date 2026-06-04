
3. Crear y activar entorno virtual
powershell
12
Deberías ver (.venv) al inicio de la línea de comandos.
4. Instalar dependencias
powershell
1
Esto instalará: YOLOv8, EasyOCR, Streamlit, OpenCV, Plotly, Pandas, PyTorch, etc.
Tiempo estimado: 5-10 minutos dependiendo de tu conexión.
5. Verificar instalación
powershell
1
Si ves el mensaje de éxito, estás listo para usar el sistema.
Cómo usar la aplicación
Ejecutar el sistema
powershell
1
Se abrirá automáticamente tu navegador en http://localhost:8501.
Qué deberías ver
Pantalla principal con dos pestañas:
Detección de Placas - Para subir imágenes y procesarlas
Panel de Análisis - Dashboard con estadísticas
Panel lateral izquierdo con:
Slider para ajustar confianza de detección (0.10 a 0.90)
Checkbox para mostrar preprocesamiento OCR
Botón para limpiar historial
Contador de detecciones registradas
Proceso de detección
Paso 1: Subir imagen
Haz clic en "Seleccione una imagen para procesar"
Formatos soportados: JPG, JPEG, PNG
Resolución recomendada: mínimo 1280x720 píxeles
Paso 2: Procesamiento automático
El sistema realizará:
Detección con YOLOv8 - Localiza la placa en la imagen
Recorte de la placa - Extrae la región detectada
Preprocesamiento OCR - Mejora la imagen (escala de grises, binarización, sharpening)
Reconocimiento de texto - Lee los caracteres con EasyOCR
Validación de formato - Verifica que sea una placa colombiana válida
Clasificación por color - Identifica tipo de vehículo (moto, carro, público, etc.)
Cálculo de Pico y Placa - Determina si tiene restricción activa
Paso 3: Ver resultados
En la columna izquierda:
Imagen original con bounding box verde alrededor de la placa detectada
Mensaje de éxito o advertencia
En la columna derecha:
Placa identificada (ej: OBW59D)
Tipo de vehículo (ej: Motocicleta Particular)
Formato (ej: Motocicleta)
Dígito de restricción (ej: 9)
Estado de Pico y Placa (ej: Libre circulación o RESTRICCIÓN ACTIVA)
Métricas de confianza de detección y OCR
Dashboard y Visualizaciones
Qué observar en el Panel de Análisis
Métricas superiores (4 tarjetas):
Total de detecciones realizadas
Número de motocicletas detectadas
Vehículos con restricción activa
Vehículos sin restricción
Visualización 1: Gráfico de Barras
Muestra cantidad de placas por tipo de vehículo
Qué observar: El tipo más frecuente y el porcentaje de motocicletas
Análisis esperado: "El tipo más detectado es Motocicleta Particular con X detecciones"
Visualización 2: Gráfico Circular
Muestra porcentaje de vehículos con/sin restricción
Qué observar: Colores rojo (restringidos) y verde (libres)
Análisis esperado: "El X% de vehículos presenta restricción activa"
Visualización 3: Gráfico de Líneas
Muestra detecciones por hora del día (0-23 horas)
Qué observar: Picos de actividad en horas 6-9 AM y 5-8 PM
Análisis esperado: "La hora con mayor actividad es las X:00 con Y detecciones"
Tabla de historial:
Registro cronológico de todas las detecciones
Columnas: Fecha/Hora, Placa, Tipo, Formato, Estado
Entrenamiento del modelo (opcional)
Si deseas reentrenar el modelo con tu propio dataset:
Preparar dataset
Organiza tus imágenes en carpetas:
123
Cada imagen debe tener su archivo de anotación .txt en formato YOLO.
Ejecutar entrenamiento
powershell
1
Parámetros configurados:
50 épocas de entrenamiento
Tamaño de imagen: 640x640
Batch size: 16
Early stopping con paciencia de 20 épocas
Tiempo estimado:
CPU: 4-8 horas
GPU NVIDIA: 30-60 minutos
Qué observar durante el entrenamiento
Progreso de épocas en la consola
Métricas: mAP50, mAP50-95, Precision, Recall
El modelo se guarda automáticamente en runs/best.pt
Métricas de referencia:
mAP50 > 0.90: Excelente
mAP50 entre 0.70 y 0.90: Aceptable
mAP50 < 0.70: Requiere más datos de entrenamiento
Formatos de placas soportados
Tipo
Formato
Ejemplo
Color
Moto particular
AAA99A
OBW59D
Amarillo
Carro tradicional
AAA999
ABC123
Amarillo
Carro Mercosur
AAA999A
ABC123D
Amarillo
Servicio público
AA999A
AB123C
Blanco
Diplomático
CD999
CD123
Azul
Remolque
R99999
R12345
Verde
Pico y Placa Bogotá
Horario: Lunes a viernes, 6:00 AM - 8:00 PM
Día
Dígitos restringidos
Lunes
4, 5, 6, 7
Martes
8, 9, 0, 1
Miércoles
2, 3, 4, 5
Jueves
6, 7, 8, 9
Viernes
0, 1, 2, 3
Motocicletas: Restricción los sábados (dígitos 0-4 restringidos, 5-9 libres)
Fines de semana: Libre circulación
Problemas comunes y soluciones
Error: "ModuleNotFoundError"
Solución: Instalar la dependencia faltante
powershell
1
El sistema no detecta placas
Posibles causas:
Confianza muy alta → Reducir slider a 0.20-0.30
Imagen borrosa o con mala iluminación → Usar foto más clara
Ángulo muy inclinado → Tomar foto más frontal
Modelo no entrenado suficientemente → Reentrenar con más datos
El OCR lee mal los caracteres
Recomendaciones:
Usar imágenes de mayor resolución
Mejorar iluminación
Activar "Mostrar preprocesamiento OCR" para depurar
Limpiar la placa si está sucia
La aplicación es lenta
Soluciones:
Usar GPU NVIDIA si está disponible
Reducir tamaño de imagen: cambiar imgsz=1280 a imgsz=640 en el código
Cerrar otras aplicaciones que consuman recursos
Estructura del proyecto
12345678910111213
Requisitos del sistema
Hardware mínimo:
Procesador: Intel i5 / AMD Ryzen 5
RAM: 8 GB (16 GB recomendado)
Almacenamiento: 5 GB libres
GPU: Opcional (NVIDIA con CUDA para aceleración)
Software:
Windows 10/11 (64-bit)
Python 3.11+
Git 2.30+
Licencia
Este proyecto se distribuye bajo la Licencia MIT.
Versión 2.0 - Junio 2026
Desarrollado como proyecto académico de visión por computadora e inteligencia artificial aplicada.