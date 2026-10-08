# 🎬 GUION TÉCNICO DE GRABACIÓN DE VIDEO (6 A 8 MINUTOS)
**Evidencia:** AA2-EV03 - Video Configuración de Analítica de Datos  
**Programa:** Aplicación de la Inteligencia Artificial en la Integración de Datos | SENA  
**Duración objetivo:** 07:00 minutos (Margen permitido: 6 a 8 minutos)  
**Herramientas para grabar:** OBS Studio (1080p, audio claro)

---

## ⏱️ ESTRUCTURA Y CRONOGRAMA MINUTO A MINUTO

```
[00:00 - 00:45] Bloque 1: Presentación y Objetivos
[00:45 - 01:45] Bloque 2: Justificación Técnica (Python + Antigravity vs Power BI)
[01:45 - 02:45] Bloque 3: Configuración del Entorno y Estructura
[02:45 - 04:00] Bloque 4: Ingesta y Limpieza Estadística de Datos
[04:00 - 05:00] Bloque 5: Análisis Exploratorio (EDA) y Segmentaciones
[05:00 - 06:15] Bloque 6: Machine Learning (K-Means) y Método del Codo
[06:15 - 07:15] Bloque 7: Integración de IA Generativa y Orquestación en Vivo
[07:15 - 07:45] Bloque 8: Conclusiones y Cierre
```

---

## 🎙️ DESGLOSE DETALLADO: QUÉ DECIR Y QUÉ MOSTRAR EN PANTALLA

### 🔹 Minuto 00:00 - 00:45 | Introducción y Presentación
* **Pantalla a mostrar:** VS Code abierto con la portada o el archivo `README.md` visible.
* **Acción:** Muestra tu rostro en cámara pequeña (opcional) o la pantalla clara de VS Code.
* **Voz / Locución:**
  > *"Cordial saludo, instructor y compañeros. Mi nombre es [Tu Nombre Completo] y en este video presento el desarrollo de la evidencia **AA2-EV03: Video Configuración de Analítica de Datos**, perteneciente al programa **Aplicación de la Inteligencia Artificial en la Integración de Datos** del SENA.*  
  > *El objetivo principal es configurar un entorno profesional de analítica de datos en Python asistido por el agente generativo **Antigravity**, procesando el dataset del Titanic mediante técnicas estadísticas rigurosas, segmentación con K-Means y un modelo de lenguaje para la interpretación automática de resultados."*

---

### 🔹 Minuto 00:45 - 01:45 | Justificación Técnica (Python vs Power BI / Tableau)
* **Pantalla a mostrar:** Documento comparativo o tabla comparativa en markdown (`README.md` sección 3).
* **Acción:** Resalta con el cursor los criterios de la tabla comparativa.
* **Voz / Locución:**
  > *"Para este proyecto seleccionamos un stack basado en **Python, VS Code y Antigravity** sobre herramientas tradicionales como Power BI o Tableau. La justificación técnica radica en tres pilares fundamentales:*  
  > *1. **Capacidad algorítmica:** Mientras las herramientas BI están limitadas a visualizaciones y cálculos DAX, Python nos brinda acceso directo al ecosistema de Scikit-Learn y SciPy para modelos avanzados de Machine Learning.*  
  > *2. **Integración nativa con IA Generativa:** Podemos conectar APIs de LLMs como OpenAI GPT-4o-mini directamente dentro del pipeline de datos, automatizando la narrativa de negocio.*  
  > *3. **Reproducibilidad y DevOps:** Todo el flujo se controla mediante código versionable en Git, entornos virtuales y dependencias declaradas en `requirements.txt`, cumpliendo con estándares de ingeniería de datos."*

---

### 🔹 Minuto 01:45 - 02:45 | Configuración del Entorno y Arquitectura
* **Pantalla a mostrar:** Terminal integrada de VS Code y el árbol de carpetas lateral.
* **Acción:** Ejecuta en la terminal `python --version` y muestra el entorno virtual activado (`(venv)`), luego expande las carpetas `data/`, `src/`, `outputs/` y `docs/`.
* **Voz / Locución:**
  > *"Aquí podemos observar el entorno de trabajo configurado. Contamos con Python 3.12 y un entorno virtual con librerías fijadas: pandas, scikit-learn, seaborn, matplotlib y openai.*  
  > *La arquitectura del proyecto sigue las mejores prácticas:*  
  > *Una carpeta `data/` separando datos crudos (`raw/`) de procesados (`processed/`), una carpeta `src/` con scripts modulares numerados según el ciclo CRISP-DM, la carpeta `outputs/` para figuras y reportes generados, y la carpeta `docs/` con el documento maestro y la bitácora de prompts."*

---

### 🔹 Minuto 02:45 - 04:00 | Ingesta y Limpieza Estadística
* **Pantalla a mostrar:** Scripts `src/01_ingesta.py` y `src/02_limpieza.py`.
* **Acción:** Abre `02_limpieza.py`, muestra la función `limpiar_titanic()` y señala los comentarios estadísticos.
* **Voz / Locución:**
  > *"En la fase de ingesta cargamos el dataset del Titanic con 891 registros y auditamos los valores nulos. Detectamos 177 nulos en 'Age', 687 en 'Cabin' y 2 en 'Embarked'.*  
  > *En la fase de limpieza aplicamos decisiones estadísticas justificadas:*  
  > *Primero, descartamos 'Cabin' porque más del 77% de los datos faltan, y descartamos 'Ticket' y 'Name' por alta dispersión.*  
  > *Segundo, para la variable 'Age' no aplicamos una mediana simple global porque generaría sesgo; en su lugar, implementamos **imputación condicional de mediana agrupada por Clase ('Pclass') y Género ('Sex')**, preservando las diferencias demográficas.*  
  > *Tercero, imputamos 'Embarked' con la moda por ser categórica y tener menos del 0.3% de nulos.*  
  > *Y finalmente, creamos la variable 'FamilySize' sumando SibSp, Parch más uno."*

---

### 🔹 Minuto 04:00 - 05:00 | EDA y Segmentaciones Clave
* **Pantalla a mostrar:** Script `src/03_eda.py` y abre las imágenes `heatmap_correlaciones.png` y `countplot_supervivencia.png` en el visor de VS Code.
* **Acción:** Muestra las figuras generadas en `outputs/figuras/` y las líneas de código donde se aplican los filtros.
* **Voz / Locución:**
  > *"En el análisis exploratorio generamos la matriz de correlaciones, el boxplot de tarifas por clase y el conteo de supervivencia por género.*  
  > *Implementamos segmentaciones con filtros que revelan patrones críticos:*  
  > *Al filtrar mujeres de primera clase, observamos una tasa de supervivencia cercana al **96.8%**, mientras que al evaluar hombres de tercera clase, la tasa desciende drásticamente a un **13.5%**.*  
  > *Esto demuestra numéricamente el impacto conjunto de los factores socioeconómicos y la política de 'mujeres y niños primero'."*

---

### 🔹 Minuto 05:00 - 06:15 | Modelado con K-Means y Método del Codo
* **Pantalla a mostrar:** `src/04_modelado.py` y las figuras `metodo_codo.png` y `scatter_clusters.png`.
* **Acción:** Abre `metodo_codo.png` y explica la curva. Luego abre `scatter_clusters.png`.
* **Voz / Locución:**
  > *"Para la segmentación no supervisada utilizamos **K-Means Clustering** sobre las variables Edad y Tarifa.*  
  > *Como K-Means se basa en distancias euclidianas, primero aplicamos **StandardScaler** para estandarizar las escalas y evitar que las tarifas dominen sobre la edad.*  
  > *Luego ejecutamos el **Método del Codo** evaluando la inercia WCSS de k=2 a k=10. Como se aprecia en la gráfica, el punto de inflexión evidente ocurre en **k=3**.*  
  > *Entrenamos el modelo con 3 clusters y aquí en el scatter plot vemos la segmentación resultante:*  
  > *Cluster 0: Pasajeros de tarifa baja y edad joven/adulta.*  
  > *Cluster 1: Pasajeros de alta gama con tarifas premium y alta supervivencia.*  
  > *Cluster 2: Adultos mayores y clase media con tarifas moderadas."*

---

### 🔹 Minuto 06:15 - 07:15 | IA Generativa y Orquestación en Vivo
* **Pantalla a mostrar:** `src/05_ia_generativa.py` y terminal ejecutando `python main.py`.
* **Acción:** Muestra el script 05, abre la terminal y ejecuta `python main.py` mostrando cómo fluye todo el pipeline en vivo. Luego abre `interpretacion_ia.txt`.
* **Voz / Locución:**
  > *"Aquí reside el valor diferencial de nuestro pipeline: la integración de **IA Generativa**. El script `05_ia_generativa.py` toma la tabla de métricas calculada por K-Means y envía un prompt estructurado a un Modelo de Lenguaje.*  
  > *Vamos a ejecutar en vivo el orquestador `main.py`...*  
  > *(Muestras la ejecución en consola y sus logs).*  
  > *Como vemos, el pipeline ejecutó en pocos segundos la ingesta, limpieza, EDA, modelado, IA generativa y dashboard consolidado.*  
  > *Y en `outputs/reportes/interpretacion_ia.txt` podemos ver cómo la IA tradujo los datos estadísticos a un informe gerencial con perfil demográfico y conclusiones operativas."*

---

### 🔹 Minuto 07:15 - 07:45 | Conclusiones y Despedida
* **Pantalla a mostrar:** Árbol de carpetas con todos los archivos generados y el PDF de soporte.
* **Acción:** Muestra el checklist cumplido al 100%.
* **Voz / Locución:**
  > *"En conclusión, hemos configurado con éxito un entorno completo de analítica de datos en Python asistido por el agente generativo Antigravity, cumpliendo a cabalidad con todos los criterios de evaluación de la evidencia AA2-EV03.*  
  > *Todos los scripts, gráficos, bitácora de prompts y el documento maestro en PDF se encuentran listos para su revisión. Muchas gracias por su atención."*

---

## 🎯 CONSEJOS CLAVE PARA OBTENER 100/100
1. **Puntualidad en el tiempo:** Mantente entre 6 minutos y 7 minutos 30 segundos.
2. **Audio:** Habla con tono seguro, pausado y técnico (menciona términos como *imputación condicional, inercia WCSS, StandardScaler, CRISP-DM*).
3. **Grabación limpia:** Cierra pestañas o notificaciones innecesarias en tu pantalla.
4. **Subida del video:** Sube a YouTube en modo **"No listado" (Unlisted)** y verifica abriendo el enlace en una pestaña de incógnito.

