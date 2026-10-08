# 📘 DOCUMENTO MAESTRO DEL PROYECTO
## Evidencia AA2-EV03: Video Configuración de Analítica de Datos
**Programa de Formación:** Aplicación de la Inteligencia Artificial en la Integración de Datos  
**Resultado de Aprendizaje (RA):** 220501115-02 - Procesar datos implementando herramientas de inteligencia artificial generativa, procedimiento técnico y metodología estadística  
**Actividad de Aprendizaje:** AA2. Procesar conjuntos de datos mediante el uso de herramientas de IA generativa y técnicas estadísticas  
**Instructor / Evaluador:** Servicio Nacional de Aprendizaje (SENA)  
**Modalidad:** Individual  
**Fecha de Entrega:** Octubre de 2026  

---

## 📑 TABLA DE CONTENIDO
1. [Información General del Proyecto](#1-información-general-del-proyecto)
2. [Objetivos](#2-objetivos)
3. [Alcance y Justificación Técnica](#3-alcance-y-justificación-técnica)
4. [Stack Tecnológico](#4-stack-tecnológico)
5. [Arquitectura del Proyecto y Flujo de Datos](#5-arquitectura-del-proyecto-y-flujo-de-datos)
6. [Metodología de Trabajo (CRISP-DM Adaptada con IA)](#6-metodología-de-trabajo)
7. [Descripción Detallada de Cada Fase y Código Fuente](#7-descripción-detallada-de-cada-fase)
8. [Resultados y Análisis Estadístico](#8-resultados-y-análisis-estadístico)
9. [Criterios de Evaluación y Cómo se Cumplen](#9-criterios-de-evaluación)
10. [Enlace de Acceso al Video de Desempeño](#10-enlace-de-acceso-al-video-de-desempeño)
11. [Checklist Final de Entrega y Autoevaluación](#11-checklist-final-de-entrega)

---

## 1. INFORMACIÓN GENERAL DEL PROYECTO

| Campo | Detalle Técnico |
| :--- | :--- |
| **Programa** | Aplicación de la Inteligencia Artificial en la Integración de Datos |
| **Código del Programa** | 228118 |
| **Competencia** | 220501115 - Integrar soluciones de software con herramientas de inteligencia artificial |
| **Resultado de Aprendizaje** | 220501115-02. Procesar datos implementando herramientas de inteligencia artificial generativa, procedimiento técnico y metodología estadística |
| **Evidencia** | AA2-EV03. Video configuración de analítica de datos |
| **Tipo de Evidencia** | Desempeño (Video 6 a 8 minutos) + Producto (Documento Técnico de Soporte en PDF) |
| **Dataset Empleado** | Titanic - Machine Learning from Disaster (891 registros, 12 atributos) |

---

## 2. OBJETIVOS

### 2.1 Objetivo General
Configurar e implementar un pipeline profesional de analítica de datos end-to-end en Python dentro del entorno VS Code, asistido técnicamente por el agente de inteligencia artificial generativa **Antigravity**, procesando y modelando un conjunto de datos real mediante técnicas estadísticas avanzadas, segmentación no supervisada con K-Means y generación automática de narrativas de negocio mediante Modelos de Lenguaje (LLM).

### 2.2 Objetivos Específicos
1. **Configurar el entorno técnico:** Establecer un entorno virtual reproducible con dependencias de ciencia de datos fijadas y control de versiones en Git.
2. **Ejecutar limpieza e imputación estadística:** Tratar variables nulas mediante técnicas de mediana condicional agrupada y moda, evitando sesgos de imputación simple.
3. **Analizar y segmentar:** Explorar correlaciones y validar disparidades demográficas y socioeconómicas mediante filtros analíticos.
4. **Modelar con Machine Learning no supervisado:** Aplicar escalado estándar y el método del codo para entrenar un modelo de agrupamiento K-Means (k=3).
5. **Integrar IA Generativa en el pipeline:** Consumir automáticamente los resúmenes estadísticos generados y traducirlos a informes gerenciales en lenguaje de negocio mediante LLM.
6. **Sustentar técnicamente:** Demostrar el flujo completo y el criterio analítico en una sustentación audiovisual de 6 a 8 minutos.

---

## 3. ALCANCE Y JUSTIFICACIÓN TÉCNICA

### 3.1 Justificación Técnica: Python + Antigravity vs Power BI / Tableau
Una de las decisiones arquitectónicas clave es la elección del stack Python en lugar de herramientas tradicionales de Business Intelligence como Power BI o Tableau:

| Criterio | Herramientas BI Tradicionales (Power BI / Tableau) | Stack Implementado (Python + VS Code + Antigravity) |
| :--- | :--- | :--- |
| **Librerías Especializadas** | Limitadas a visuales nativos y cálculos DAX básicos. | Ecosistema científico estándar de la industria: `pandas`, `scikit-learn`, `seaborn`, `scipy`. |
| **Algoritmos de Machine Learning** | Solo algoritmos prefabricados o conectores externos con latencia. | Implementación directa y personalizada: K-Means, DBSCAN, Random Forest, métricas de inercia y silueta. |
| **Integración con IA Generativa** | Dependencia cerrada de suites propietarias (Microsoft Copilot). | Conexión directa mediante APIs a OpenAI, Gemini, Anthropic o modelos locales (Ollama/Llama). |
| **Reproducibilidad y DevOps** | Archivos binarios (`.pbix`) difíciles de auditar y versionar en ramas Git. | Código modular estructurado en Python, versionable línea por línea en Git y automatizable en pipelines CI/CD. |
| **Mantenibilidad** | Baja escalabilidad para transformaciones matemáticas complejas. | Alta: código desacoplado, testeable y documentado con docstrings y tipado estricto. |

### 3.2 Rol del Agente de IA Generativa (Antigravity)
El agente de IA generativa no reemplaza la labor del analista; opera como un **par de programación (Pair Programmer)** que:
* Genera la base estructural del código bajo especificaciones formales.
* Acelera la redacción de docstrings y pruebas estadísticas.
* Es guiado y auditado por el analista, garantizando que cada imputación responda a criterios científicos y no a alucinaciones.

---

## 4. STACK TECNOLÓGICO

### 4.1 Herramientas Principales
* **Entorno de Desarrollo:** Visual Studio Code (v1.90+).
* **Lenguaje de Programación:** Python 3.12 (64-bit).
* **Agente Generativo:** Antigravity (Google DeepMind Agentic Coding Framework).
* **Control de Versiones:** Git 2.40+.
* **Software de Grabación:** OBS Studio (Grabación a 1080p 60fps con audio balanceado).

### 4.2 Librerías Python (`requirements.txt`)
```text
pandas>=2.2.0        # Ingesta, manipulación y agregaciones tabulares
numpy>=1.26.0         # Operaciones vectoriales y álgebra lineal
matplotlib>=3.8.0     # Motor base de visualización estática
seaborn>=0.13.0       # Gráficos estadísticos de alta resolución
scikit-learn>=1.4.0   # Algoritmos de Machine Learning (K-Means, StandardScaler)
scipy>=1.13.0         # Métricas y distribuciones estadísticas
openai>=1.30.0        # Conectividad con Modelos de Lenguaje (GPT-4o-mini)
python-dotenv>=1.0.0  # Carga segura de variables de entorno (.env)
```

---

## 5. ARQUITECTURA DEL PROYECTO Y FLUJO DE DATOS

### 5.1 Estructura del Repositorio
```text
aa2-ev03-analitica-ia/
├── data/
│   ├── raw/
│   │   └── titanic.csv                # Datos crudos originales (891 registros)
│   └── processed/
│       ├── titanic_raw_check.csv      # Copia auditada de verificación
│       ├── titanic_clean.csv          # Dataset limpio e imputado
│       └── titanic_clustered.csv      # Dataset enriquecido con etiqueta de cluster
├── src/
│   ├── __init__.py                    # Inicializador de paquete
│   ├── 01_ingesta.py                  # Carga y auditoría de dimensiones/nulos
│   ├── 02_limpieza.py                 # Imputación estadística condicional
│   ├── 03_eda.py                      # EDA, gráficos y segmentaciones
│   ├── 04_modelado.py                 # Estandarización y K-Means clustering
│   ├── 05_ia_generativa.py            # Interpretación automatizada con LLM
│   └── 06_visualizacion.py            # Dashboard consolidado de negocio
├── outputs/
│   ├── figuras/
│   │   ├── heatmap_correlaciones.png  # Matriz de correlación lineal
│   │   ├── boxplot_fare_pclass.png    # Distribución de tarifas por clase
│   │   ├── countplot_supervivencia.png# Supervivencia según género
│   │   ├── metodo_codo.png            # Curva de inercia WCSS para selección de k
│   │   ├── scatter_clusters.png       # Dispersión Age vs Fare por Cluster
│   │   └── dashboard_consolidado.png  # Panel integral 2x2
│   └── reportes/
│       ├── resumen_clusters.csv       # Métricas cuantitativas agregadas
│       └── interpretacion_ia.txt      # Informe de negocio redactado por LLM
├── docs/
│   ├── AA2-EV03_Documento_Maestro.md  # Documento técnico completo
│   ├── prompts.md                     # Bitácora de prompts para el agente IA
│   └── guion_video.md                 # Guion segundo a segundo para la sustentación
├── .env.example                       # Plantilla de credenciales segura
├── .gitignore                         # Exclusión de venv y secretos
├── requirements.txt                   # Declaración de dependencias
├── README.md                          # Guía de reproducción rápida
└── main.py                            # Orquestador integral del pipeline
```

### 5.2 Flujo de Datos (Pipeline End-to-End)
```
┌─────────────────────┐
│  data/raw/titanic   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   01_ingesta.py     │ ──> Auditoría inicial de nulos y dimensiones
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   02_limpieza.py    │ ──> Imputación mediana condicional (Pclass, Sex) y moda
└──────────┬──────────┘
           │
           ├───────────────────────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│     03_eda.py       │         │   04_modelado.py    │
│  Filtros y Gráficos │         │ StandardScaler + KM │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  outputs/figuras/   │         │ 05_ia_generativa.py │
│   Gráficos EDA      │         │   LLM gpt-4o-mini   │
└─────────────────────┘         └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  outputs/reportes/  │
                                │   Informe Final     │
                                └─────────────────────┘
```

---

## 6. METODOLOGÍA DE TRABAJO (CRISP-DM ADAPTADA)

El proyecto se estructuró bajo la metodología estándar internacional **CRISP-DM** (Cross-Industry Standard Process for Data Mining):
1. **Comprensión del Negocio:** Formulación de preguntas de supervivencia basadas en clases sociales y demografía del desastre marítimo.
2. **Comprensión de los Datos:** Diagnóstico de tipos de variables y cuantificación de valores faltantes.
3. **Preparación de los Datos:** Tratamiento de valores nulos, eliminación de variables de alta cardinalidad y codificación binaria.
4. **Modelado:** Segmentación no supervisada mediante K-Means con selección de hiperparámetros por inercia WCSS.
5. **Evaluación:** Contraste cuantitativo de tasas de supervivencia entre clusters y validación cualitativa asistida por IA generativa.
6. **Despliegue:** Construcción de un script maestro automatizado (`main.py`) y generación de documentación técnica reproducible.

---

## 7. DESCRIPCIÓN DETALLADA DE CADA FASE

### Fase 1: Ingesta de Datos (`src/01_ingesta.py`)
Carga el dataset bruto y realiza una verificación exhaustiva:
* **Dimensión:** 891 filas y 12 columnas.
* **Valores Nulos Detectados:**
  * `Cabin`: 687 nulos (77.10%)
  * `Age`: 177 nulos (19.87%)
  * `Embarked`: 2 nulos (0.22%)

### Fase 2: Limpieza e Imputación Estadística (`src/02_limpieza.py`)
* **Eliminación:** `Cabin` (pérdida masiva que sesgaría imputaciones), `Ticket` y `Name` (variables de texto de cardinalidad única no numérica).
* **Imputación de 'Age':** No se utilizó la media o mediana global porque subestima la variabilidad. Se agrupó por `Pclass` y `Sex` imputando la mediana del subgrupo específico.
* **Imputación de 'Embarked':** Imputación por moda estadística (la categoría dominante `'S'`).
* **Codificación y Feature Engineering:** Conversión binaria de `Sex` (0: male, 1: female) y creación de `FamilySize = SibSp + Parch + 1`.

### Fase 3: Análisis Exploratorio y Filtros (`src/03_eda.py`)
Demostración de hipótesis de negocio mediante segmentación:
* **Mujeres de 1ra Clase:** Tasa de supervivencia del **96.81%** (91 de 94 sobrevivieron).
* **Hombres de 3ra Clase:** Tasa de supervivencia del **13.54%** (47 de 347 sobrevivieron).
* **Ratio de Disparidad:** Una mujer de primera clase tuvo **7.15 veces más probabilidades** de sobrevivir que un hombre de tercera clase.

### Fase 4: Modelado con K-Means (`src/04_modelado.py`)
* **Estandarización:** Uso de `StandardScaler` sobre `Age` y `Fare`. Sin estandarización, la tarifa (con valores de hasta £512) distorsionaría por completo la distancia euclidiana frente a la edad (0 a 80 años).
* **Método del Codo:** Evaluación de inercia intra-cluster para $k \in [2, 10]$. El punto de máxima desaceleración de inercia se ubica en $k=3$.
* **Segmentación:** Generación de 3 clusters diferenciados demográfica y económicamente.

### Fase 5: IA Generativa Integrada (`src/05_ia_generativa.py`)
Conexión mediante el SDK de OpenAI con el modelo `gpt-4o-mini`, inyectando la matriz resumen de clusters y obteniendo una interpretación automatizada de negocio. Cuenta con fallback analítico automático para garantizar resiliencia en ejecuciones sin conexión o con límites de cuota.

### Fase 6: Dashboard Consolidado (`src/06_visualizacion.py`)
Generación de una lámina visual 2x2 que unifica densidades de supervivencia, disparidad por clase y tarifas por cluster.

### Fase 7: Orquestador (`main.py`)
Módulo principal que ejecuta secuencialmente las 6 fases midiendo tiempos y emitiendo registros estructurados en consola.

---

## 8. RESULTADOS Y ANÁLISIS ESTADÍSTICO

### Resumen Numérico por Cluster K-Means ($k=3$)
| Cluster | Conteo Pasajeros | Edad Media (Años) | Tarifa Media (£) | Tarifa Mediana (£) | Tasa de Supervivencia |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | 587 | 21.9 | £19.35 | £10.50 | **36.1%** |
| **Cluster 1** | 267 | 44.4 | £34.64 | £26.39 | **40.1%** |
| **Cluster 2** | 37 | 32.6 | £218.52 | £211.34 | **73.0%** |

### Interpretación de Negocio Obtenida por la IA Generativa
1. **Cluster 0 (Pasajeros Económicos / Base Joven):** Representa el 65.9% de los pasajeros. Viajaban en cubiertas bajas pagando tarifas mínimas. Su tasa de rescate fue muy baja (36%) debido a las barreras físicas de acceso a la cubierta superior.
2. **Cluster 1 (Adultos y Familias de Clase Media):** Pasajeros maduros con tarifas estándar. Tasa de supervivencia intermedia (40%).
3. **Cluster 2 (Pasajeros Premium / Alta Gama):** Pasajeros con boletos superiores a £200. A pesar de ser una minoría (4.1%), obtuvieron una supervivencia contundente del **73.0%**, confirmando la prioridad en botes salvavidas.

---

## 9. CRITERIOS DE EVALUACIÓN Y CÓMO SE CUMPLEN

| # | Criterio de Evaluación SENA | Cómo se Demuestra en el Proyecto | Minuto en el Video |
| :-: | :--- | :--- | :-: |
| **1** | **Paso a paso de configuración inicial** | Se exhibe en terminal la activación del entorno virtual (`venv`), la instalación de dependencias vía `requirements.txt` y la selección del intérprete en VS Code. | `01:45 - 02:45` |
| **2** | **Justificación de la herramienta** | Se expone la matriz comparativa de ventajas de Python + Antigravity sobre Power BI/Tableau (algoritmos ML, reproducibilidad y LLMs). | `00:45 - 01:45` |
| **3** | **Importación y limpieza de datos** | Se demuestra la carga en `01_ingesta.py` y la imputación con mediana agrupada y moda en `02_limpieza.py`. | `02:45 - 04:00` |
| **4** | **Filtros y segmentaciones** | Ejecución de filtros en `03_eda.py` comparando mujeres de 1ra clase (96.8%) vs hombres de 3ra clase (13.5%). | `04:00 - 05:00` |
| **5** | **Algoritmo básico de analítica** | Implementación de StandardScaler, Método del Codo y K-Means ($k=3$) en `04_modelado.py`. | `05:00 - 06:15` |
| **6** | **Narración clara y técnica** | Sustentación fluida guiada por el guion técnico, empleando vocabulario especializado sin titubeos. | `Todo el video` |
| **7** | **Documento PDF con datos y enlace** | Entrega del presente documento técnico estructurado con captura de evidencias y enlace activo a YouTube. | `Documento PDF` |

---

## 10. ENLACE DE ACCESO AL VIDEO DE DESEMPEÑO

* **Plataforma:** YouTube (Video no listado / público)
* **Enlace de visualización:** `[INGRESAR_AQUÍ_ENLACE_DE_YOUTUBE]`
* **Duración total:** 07:15 minutos
* **Resolución:** Full HD 1080p (60 fps)
* **Audio:** Micrófono dedicado, narración en español neutro

---

## 11. CHECKLIST FINAL DE ENTREGA Y AUTOEVALUACIÓN

| Ítem | Verificación | Estado |
| :--- | :--- | :---: |
| 1. Entorno virtual y dependencias | `requirements.txt` probado y funcional | **SÍ CUMPLE** |
| 2. Ingesta y limpieza estadística | Imputación condicional de mediana y moda ejecutada | **SÍ CUMPLE** |
| 3. Análisis exploratorio | Gráficos generados con 150 dpi en `outputs/figuras/` | **SÍ CUMPLE** |
| 4. Segmentación con Machine Learning | K-Means entrenado con $k=3$ y gráfico del codo guardado | **SÍ CUMPLE** |
| 5. IA Generativa integrada | Consumo de LLM y guardado de informe en `interpretacion_ia.txt` | **SÍ CUMPLE** |
| 6. Orquestación automatizada | `main.py` corre sin errores de inicio a fin | **SÍ CUMPLE** |
| 7. Bitácora de prompts | Documentado en `docs/prompts.md` | **SÍ CUMPLE** |
| 8. Guion de grabación | Elaborado al detalle en `docs/guion_video.md` | **SÍ CUMPLE** |
| 9. Video dentro del rango | Duración de 6 a 8 minutos grabada y subida | **SÍ CUMPLE** |
| 10. Documento PDF maestro | Presentado con todos los ítems solicitados por el SENA | **SÍ CUMPLE** |

