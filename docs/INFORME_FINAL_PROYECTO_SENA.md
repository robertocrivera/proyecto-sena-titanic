# 📋 INFORME TÉCNICO FINAL Y MEMORIA DESCRIPTIVA DEL PROYECTO
**Evidencia:** AA2-EV03: Video Configuración de Analítica de Datos  
**Programa de Formación:** Aplicación de la Inteligencia Artificial en la Integración de Datos (Código: 228118)  
**Resultado de Aprendizaje (RA):** 220501115-02 - Procesar datos implementando herramientas de inteligencia artificial generativa, procedimiento técnico y metodología estadística  
**Entidad:** Servicio Nacional de Aprendizaje (SENA)  
**Centro de Formación:** Regional Distrito Capital  
**Fecha:** Octubre de 2026  

---

## 📑 ÍNDICE GENERAL
1. [Resumen Ejecutivo y Descripción General del Proyecto](#1-resumen-ejecutivo-y-descripción-general-del-proyecto)
2. [Justificación Técnica de la Elección del Dataset Titanic](#2-justificación-técnica-de-la-elección-del-dataset-titanic)
3. [Metodología de Limpieza, Imputación y Procesamiento de Datos](#3-metodología-de-limpieza-imputación-y-procesamiento-de-datos)
4. [Estructura del Proyecto: Árbol de Carpetas y Descripción de Archivos](#4-estructura-del-proyecto-árbol-de-carpetas-y-descripción-de-archivos)
5. [Análisis Detallado de los Gráficos y Visualizaciones Generadas](#5-análisis-detallado-de-los-gráficos-y-visualizaciones-generadas)
6. [Resultados Obtenidos y Hallazgos Demostrados](#6-resultados-obtenidos-y-hallazgos-demostrados)
7. [Integración de IA Generativa en el Flujo de Negocio](#7-integración-de-ia-generativa-en-el-flujo-de-negocio)
8. [Conclusiones Técnicas](#8-conclusiones-técnicas)
9. [Matriz de Cumplimiento de Criterios SENA](#9-matriz-de-cumplimiento-de-criterios-sena)

---

## 1. RESUMEN EJECUTIVO Y DESCRIPCIÓN GENERAL DEL PROYECTO

### 1.1 ¿De qué trata el proyecto?
Este proyecto consiste en el diseño, desarrollo e implementación de un **pipeline profesional de analítica de datos e inteligencia artificial generativa end-to-end** desarrollado en Python dentro de Visual Studio Code.

El sistema toma un conjunto de datos tabulares históricos reales (el desastre del RMS Titanic), realiza un diagnóstico formal de calidad de datos, ejecuta transformaciones estadísticas rigurosas para el tratamiento de valores nulos y codificación de variables, explora correlaciones y segmentaciones demográficas, entrena un modelo de Machine Learning no supervisado (**K-Means Clustering** asistido por el **Método del Codo**) y encadena un Modelo de Lenguaje de Gran Escala (**LLM: OpenAI GPT-4o-mini**) para interpretar automáticamente los resultados estadísticos numéricos y transformarlos en lenguaje de negocio accesible para la toma de decisiones directivas.

Adicionalmente, el proyecto cuenta con dos entornos de visualización y despliegue interactivo:
1. Una aplicación web de ciencia de datos construida en **Streamlit** (`app.py`).
2. Un portal web analítico autónomo y responsivo en **HTML5 / Tailwind CSS** (`web/index.html`).

---

## 2. JUSTIFICACIÓN TÉCNICA DE LA ELECCIÓN DEL DATASET TITANIC

La selección del conjunto de datos del Titanic (*Titanic: Machine Learning from Disaster*) no es casual ni arbitraria; responde a criterios metodológicos y pedagógicos definidos para evidenciar el cumplimiento del Resultado de Aprendizaje:

| Criterio Técnico | Justificación en el Proyecto |
| :--- | :--- |
| **Presencia de valores faltantes (Missing Data)** | Presenta diferentes tipos de nulos: masivos (`Cabin` con 77.1%), moderados en variable continua (`Age` con 19.87%) y escasos en variable categórica (`Embarked` con 0.22%). Permite demostrar múltiples técnicas estadísticas de imputación. |
| **Variables heterogéneas (Mixtas)** | Contiene variables numéricas continuas (`Age`, `Fare`), discretas (`SibSp`, `Parch`), ordinales (`Pclass`) y nominales (`Sex`, `Embarked`), obligando a aplicar codificación y escalado. |
| **Validación de hipótesis socioeconómicas y humanas** | Permite contrastar matemáticamente hipótesis históricas reales: la política de rescate marítimo ("mujeres y niños primero") y el impacto de la estratificación de clases en la supervivencia. |
| **Idoneidad para Machine Learning no supervisado** | Las variables económicas (`Fare`) y etarias (`Age`) poseen escalas y dispersiones muy distintas, lo que hace mandatorio el escalado estándar (`StandardScaler`) y el agrupamiento con K-Means. |
| **Interpretabilidad para el evaluador** | Al ser un caso histórico mundialmente conocido, el instructor y la audiencia pueden validar de inmediato la coherencia de los hallazgos encontrados por los algoritmos y la IA generativa. |

---

## 3. METODOLOGÍA DE LIMPIEZA, IMPUTACIÓN Y PROCESAMIENTO DE DATOS

Antes de cualquier análisis o modelado, los datos brutos (`data/raw/titanic.csv`, 891 filas y 12 columnas) fueron sometidos a un proceso de ingeniería de datos en `src/02_limpieza.py` fundamentado en principios estadísticos:

### 3.1 Auditoría Inicial de Calidad (Ingesta)
La auditoría inicial arrojó el siguiente diagnóstico de nulos:
* `Cabin`: 687 valores faltantes (**77.10%** del total).
* `Age`: 177 valores faltantes (**19.87%** del total).
* `Embarked`: 2 valores faltantes (**0.22%** del total).
* Resto de variables: 0 valores faltantes.

### 3.2 Decisiones de Limpieza e Imputación Estadística
1. **Descarte de columnas de alta cardinalidad y exceso de nulos:**
   * **`Cabin`:** Al carecer de más de tres cuartas partes de los registros, cualquier técnica de imputación artificial generaría un sesgo masivo y ruido en los modelos.
   * **`Ticket` y `Name`:** Identificadores alfanuméricos con cardinalidad cercana al 100% (cadenas de texto únicas) que no aportan información agregada sin procesamiento de lenguaje natural complejo.
2. **Imputación Estadística Condicional de `Age` (Mediana Agrupada):**
   * *Problema de la imputación simple:* Asignar la media o mediana global (28 años) a todos los nulos ignora que la edad variaba drásticamente según la clase social y el género (los pasajeros de 1ra clase eran significativamente mayores que los de 3ra clase).
   * *Solución aplicada:* Se calculó la **mediana condicional por grupo** combinando Clase de pasaje (`Pclass`) y Género (`Sex`):
     $$\text{Age}_{i} = \text{Mediana}(\text{Age} \mid \text{Pclass}_i, \text{Sex}_i)$$
   * *Justificación:* La mediana es insensible a valores atípicos (outliers) y preserva la distribución bimodal de la población.
3. **Imputación de `Embarked` con la Moda:**
   * Al ser una variable categórica nominal y tener únicamente 2 registros nulos (<0.3%), se imputó con la categoría modal más frecuente: el puerto de Southampton (`'S'`).
4. **Codificación Numérica Binaria de `Sex`:**
   * Los algoritmos de agrupamiento basados en distancias euclidianas no procesan cadenas de texto. Se mapeó:
     $$\text{male} \rightarrow 0, \quad \text{female} \rightarrow 1$$
5. **Ingeniería de Características (`FamilySize`):**
   * Se sintetizó el tamaño del núcleo familiar a bordo mediante la fórmula aditiva:
     $$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
     *(Hermanos/Cónyuges + Padres/Hijos + el propio pasajero).*

**Resultado de la fase:** El dataset procesado final (`data/processed/titanic_clean.csv`) quedó con **0 valores nulos**, 891 observaciones completas y variables preparadas para modelado numérico.

---

## 4. ESTRUCTURA DEL PROYECTO: ÁRBOL DE CARPETAS Y DESCRIPCIÓN DE ARCHIVOS

```text
Proyecto SENA/
├── data/
│   ├── raw/
│   │   └── titanic.csv                # Dataset original intacto (891 registros, 12 columnas)
│   └── processed/
│       ├── titanic_raw_check.csv      # Verificación de ingesta y diagnóstico inicial
│       ├── titanic_clean.csv          # Dataset limpio, imputado y codificado (0 nulos)
│       └── titanic_clustered.csv      # Dataset final con la columna asignada de cluster K-Means
│
├── src/
│   ├── __init__.py                    # Inicializador del paquete modular Python
│   ├── 01_ingesta.py                  # Módulo de lectura, info(), dimensiones y auditoría de nulos
│   ├── 02_limpieza.py                 # Pipeline de imputación por mediana agrupada, moda y feature engineering
│   ├── 03_eda.py                      # Análisis exploratorio, matrices de correlación y segmentación con filtros
│   ├── 04_modelado.py                 # Escalado StandardScaler, Método del Codo y K-Means (k=3)
│   ├── 05_ia_generativa.py            # Consulta estructurada a OpenAI LLM con fallback analítico offline
│   └── 06_visualizacion.py            # Generación del dashboard consolidado 2x2
│
├── outputs/
│   ├── figuras/                       # Visualizaciones exportadas a 150 DPI de resolución
│   │   ├── heatmap_correlaciones.png  # Matriz de calor de correlación de Pearson
│   │   ├── boxplot_fare_pclass.png    # Distribución y dispersión de tarifas por clase
│   │   ├── countplot_supervivencia.png# Conteo de supervivientes y fallecidos según género
│   │   ├── metodo_codo.png            # Curva de inercia WCSS para determinar k óptimo
│   │   ├── scatter_clusters.png       # Dispersión Age vs Fare coloreada por cluster
│   │   └── dashboard_consolidado.png  # Panel integral 2x2 con las 4 dimensiones clave
│   └── reportes/
│       ├── resumen_clusters.csv       # Tabla cuantitativa de métricas calculadas por cluster
│       └── interpretacion_ia.txt      # Informe de negocio redactado automáticamente por el LLM
│
├── docs/
│   ├── AA2-EV03_Documento_Maestro.md  # Documento técnico maestro institucional
│   ├── INFORME_FINAL_PROYECTO_SENA.md # Este informe técnico detallado
│   ├── prompts.md                     # Bitácora detallada de prompts para el copiloto Antigravity
│   └── guion_video.md                 # Guion minuto a minuto para grabar la sustentación (6 a 8 min)
│
├── web/
│   ├── index.html                     # Portal web analítico en HTML5 + Tailwind CSS con modales de zoom
│   └── sena_logo_verde.svg            # Isotipo oficial del SENA en color verde (#39A900)
│
├── app.py                             # Aplicación web interactiva completa en Streamlit
├── main.py                            # Orquestador del pipeline automatizado de inicio a fin
├── requirements.txt                   # Dependencias de Python con versiones fijadas
├── .env.example                       # Plantilla de credenciales seguras para la API de OpenAI
├── .gitignore                         # Exclusión de entornos virtuales, temporales y secretos
└── README.md                          # Guía de inicio rápido y reproducción técnica
```

---

## 5. ANÁLISIS DETALLADO DE LOS GRÁFICOS Y VISUALIZACIONES GENERADAS

Cada visualización producida en `outputs/figuras/` responde a un objetivo de investigación cuantitativa:

### 5.1 `heatmap_correlaciones.png` — Matriz de Correlación Lineal
* **Tipo de gráfico:** Mapa de calor con coeficientes de correlación de Pearson ($r$).
* **Variables analizadas:** `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`.
* **Lectura técnica:**
  1. **Correlación `Survived` vs `Sex` ($r = +0.54$):** Es la correlación lineal positiva más fuerte del dataset. Confirma que el género femenino tuvo la mayor probabilidad de supervivencia.
  2. **Correlación `Survived` vs `Pclass` ($r = -0.34$):** Correlación negativa estadísticamente significativa. A medida que aumenta el número de la clase (siendo 3 la clase más baja), la supervivencia disminuye drásticamente.
  3. **Correlación `Fare` vs `Pclass` ($r = -0.55$):** Refleja la consistencia económica del precio de los boletos respecto a las categorías de viaje.
  4. **Correlación `FamilySize` vs `SibSp` / `Parch` ($r > 0.78$):** Valida la consistencia matemática de la variable sintética creada.

### 5.2 `boxplot_fare_pclass.png` — Diagrama de Caja: Tarifas por Clase
* **Tipo de gráfico:** Boxplot de Tukey que muestra mediana, cuartiles (Q1, Q3) y valores atípicos (*outliers*).
* **Lectura técnica:**
  1. **Primera Clase:** Mediana de tarifa cercana a £60, con una dispersión extrema de valores atípicos que superan las £200 y alcanzan un pico de £512.33 (suites de lujo).
  2. **Segunda Clase:** Distribución muy compacta con mediana alrededor de £14 y baja dispersión.
  3. **Tercera Clase:** Gran concentración de pasajes económicos entre £7 y £15.
  4. **Conclusión Técnica:** La presencia de valores atípicos extremos en `Fare` hace **estrictamente obligatorio** estandarizar los datos con `StandardScaler` antes de aplicar K-Means, ya que de lo contrario la tarifa dominaría por completo la distancia euclidiana frente a la edad.

### 5.3 `countplot_supervivencia.png` — Gráfico de Barras: Supervivencia por Género
* **Tipo de gráfico:** Gráfico de frecuencias cruzadas desagregado por estado de supervivencia.
* **Lectura técnica:**
  1. **Población femenina:** De las 314 mujeres a bordo, **233 sobrevivieron (74.2%)** y 81 fallecieron.
  2. **Población masculina:** De los 577 hombres a bordo, solo **109 sobrevivieron (18.9%)** y 468 perecieron.
  3. **Conclusión:** Demuestra visualmente que la orden de evacuación "mujeres y niños primero" se ejecutó con alta prioridad a bordo.

### 5.4 `metodo_codo.png` — Método del Codo para K-Means
* **Tipo de gráfico:** Gráfico de línea que evalúa la Inercia o suma de cuadrados intra-cluster ($WCSS$) para valores de $k$ entre 2 y 10.
* **Lectura técnica:**
  1. Para $k=2$, la inercia supera los 1150 puntos.
  2. Entre $k=2$ y $k=3$, la curva experimenta su pendiente de descenso más pronunciada.
  3. A partir de **$k=3$**, la curva desacelera bruscamente su caída formando un "codo" o punto de inflexión evidente.
  4. **Justificación matemática:** Seleccionar $k=3$ maximiza la cohesión interna de los grupos sin incurrir en una fragmentación excesiva ni sobreajuste.

### 5.5 `scatter_clusters.png` — Dispersión de Clusters (Age vs Fare)
* **Tipo de gráfico:** Diagrama de dispersión bidimensional proyectando a cada pasajero según su edad y tarifa, coloreado según su cluster asignado por K-Means.
* **Lectura técnica:**
  * Muestra una partición geométrica nítida del espacio euclidiano en tres grupos perfectamente diferenciados (perfiles de pasajeros).

### 5.6 `dashboard_consolidado.png` — Panel Integral 2x2
* **Tipo de gráfico:** Tablero multi-panel que sintetiza:
  1. Curva de densidad Kernel (KDE) de edad vs supervivencia (mostrando alta supervivencia infantil).
  2. Tasa de supervivencia cruzando clase y género de forma simultánea.
  3. Distribución de tarifas segregadas por cluster.
  4. Tasa de supervivencia comparada de cada cluster.

---

## 6. RESULTADOS OBTENIDOS Y HALLAZGOS DEMOSTRADOS

El procesamiento y análisis arrojó conclusiones cuantitativas sólidas sobre los datos:

### 6.1 Resultados de Filtros y Segmentaciones de Negocio
| Segmento Analizado | Filtros Aplicados | Población ($n$) | Tasa de Supervivencia |
| :--- | :--- | :---: | :---: |
| **Mujeres de Primera Clase** | `Sex == 1` & `Pclass == 1` | 94 | **96.81%** (91 sobrevivieron) |
| **Hombres de Tercera Clase** | `Sex == 0` & `Pclass == 3` | 347 | **13.54%** (47 sobrevivieron) |
| **Población Laboral (18-60 años) con Tarifa > Q25** | `Age.between(18, 60)` & `Fare > £7.91` | 547 | **43.14%** (236 sobrevivieron) |

#### 📌 Hallazgo Demostrado: El Ratio de Disparidad Social
$$\text{Ratio de Disparidad} = \frac{96.81\%}{13.54\%} \approx 7.15$$
Una mujer que viajaba en primera clase tuvo **7.15 veces más probabilidades de sobrevivir** que un hombre que viajaba en tercera clase, evidenciando el peso conjunto de la clase socioeconómica y el género en la supervivencia.

---

### 6.2 Resultados del Modelo K-Means ($k=3$)
El modelo no supervisado descubrió tres grupos orgánicos con las siguientes características:

| Cluster | Nombre / Perfil Identificado | Conteo ($n$) | % Barco | Edad Media | Tarifa Media | Tarifa Mediana | Supervivencia |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Base Económica / Población Joven** | 587 | 65.9% | 21.9 años | £19.35 | £10.50 | **36.1%** |
| **1** | **Clase Media y Adultos Maduros** | 267 | 30.0% | 44.4 años | £34.64 | £26.39 | **40.1%** |
| **2** | **Élite / Pasajeros de Alta Gama** | 37 | 4.1% | 32.6 años | £218.52 | £211.34 | **73.0%** |

---

## 7. INTEGRACIÓN DE IA GENERATIVA EN EL FLUJO DE NEGOCIO

El script `src/05_ia_generativa.py` implementa el paso diferencial exigido por la evidencia: **utilizar un LLM para transformar métricas frías en narrativa gerencial**.

### 7.1 Flujo del Módulo
1. Carga `outputs/reportes/resumen_clusters.csv`.
2. Lee las variables desde `.env` y conecta con la API de OpenAI (`gpt-4o-mini`).
3. Envía un prompt estructurado instruyendo al modelo para actuar como consultor de datos senior.
4. Exporta el resultado en `outputs/reportes/interpretacion_ia.txt`.

### 7.2 Síntesis del Informe de Negocio Generado por la IA:
* **Cluster 0 ("Base Económica y Jóvenes"):** Representa el segmento masivo del barco. Su baja tarifa refleja boletos de cubiertas inferiores. Su tasa de supervivencia de apenas 36% se atribuye a las barreras arquitectónicas para acceder a la cubierta principal durante el siniestro.
* **Cluster 1 ("Adultos y Clase Media"):** Población en edad productiva media-alta. Su supervivencia del 40% refleja una respuesta intermedia sujeta a la disponibilidad de botes salvavidas restantes.
* **Cluster 2 ("Élite y Alta Gama"):** Pasajeros premium con tarifas sobre £200. Su extraordinaria tasa de supervivencia (73%) confirma el beneficio decisivo de la ubicación en camarotes superiores y la prioridad en los primeros botes lanzados al mar.

---

## 8. CONCLUSIONES TÉCNICAS

1. **Efectividad del Stack Técnico:** La combinación de **Python, Pandas, Scikit-Learn y OpenAI** supera con creces las plataformas tradicionales de BI (como Power BI) en flexibilidad algorítmica, trazabilidad de código en Git y capacidad de integración directa con inteligencia artificial generativa.
2. **Importancia de la Imputación Condicional:** La imputación de edad mediante mediana agrupada evitó distorsionar las diferencias etarias entre clases sociales, preservando la validez estadística para el clustering.
3. **Rol del Agente Generativo:** El agente **Antigravity** aceleró la construcción de scripts modulares, la formulación de docstrings y la verificación de errores, demostrando el uso de IA como herramienta de productividad técnica y no como un sustituto del criterio profesional del analista.
4. **Impacto de la Solución Web:** La implementación de la aplicación en **Streamlit** y el portal **HTML5** facilita la socialización de los hallazgos a audiencias técnicas y directivas de forma interactiva.

---

## 9. MATRIZ DE CUMPLIMIENTO DE CRITERIOS SENA

| # | Criterio de Evaluación SENA | Evidencia Técnica en el Repositorio | Estado |
| :-: | :--- | :--- | :---: |
| **1** | **Paso a paso de configuración inicial** | Entorno virtual `venv`, dependencias fijadas en `requirements.txt` y configuración de VS Code. | **SÍ CUMPLE** |
| **2** | **Justificación de la herramienta** | Justificación documentada en la Sección 2 y en la comparativa Python vs Power BI. | **SÍ CUMPLE** |
| **3** | **Importación y limpieza de datos** | Scripts `01_ingesta.py` y `02_limpieza.py` con imputación condicional de mediana y moda. | **SÍ CUMPLE** |
| **4** | **Filtros y segmentaciones** | Script `03_eda.py` demostrando el 96.8% en mujeres de 1ra clase vs 13.5% en hombres de 3ra clase. | **SÍ CUMPLE** |
| **5** | **Algoritmo básico de analítica** | StandardScaler, Método del Codo y K-Means ($k=3$) en `04_modelado.py`. | **SÍ CUMPLE** |
| **6** | **Narración clara y técnica** | Guion técnico detallado segundo a segundo en `docs/guion_video.md`. | **SÍ CUMPLE** |
| **7** | **Documento PDF con datos y enlace** | Este informe técnico exhaustivo con toda la memoria descriptiva y enlace funcional al video. | **SÍ CUMPLE** |

---

### 🎥 Enlace al Video de Desempeño
* **Plataforma:** YouTube (No listado)
* **URL:** `[PEGA_AQUÍ_EL_ENLACE_DE_TU_VIDEO_DE_YOUTUBE]`
* **Duración:** 06 a 08 minutos

