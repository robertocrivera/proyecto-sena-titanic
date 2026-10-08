# 🚀 AA2-EV03: Video Configuración de Analítica de Datos con IA Generativa
**Programa:** Aplicación de la Inteligencia Artificial en la Integración de Datos (SENA)  
**Resultado de Aprendizaje (RA):** 220501115-02  
**Agente de IA Generativa:** Antigravity (VS Code Assistant)  
**Dataset:** Titanic - Machine Learning from Disaster (891 filas, 12 columnas)

---

## 📋 Descripción del Proyecto
Este proyecto implementa un pipeline integral de analítica de datos end-to-end en Python asistido por el agente generativo Antigravity, cubriendo:
1. Ingesta de datos crudos y auditoría de valores nulos.
2. Limpieza e imputación estadística rigurosa (mediana condicional por clase/sexo, moda y feature engineering).
3. Análisis Exploratorio de Datos (EDA) con matrices de correlación y segmentaciones demográficas y socioeconómicas.
4. Modelado no supervisado de Machine Learning mediante **K-Means clustering** y selección de $k$ por el **Método del Codo**.
5. Integración directa de un Modelo de Lenguaje (**OpenAI GPT-4o-mini**) en el pipeline para interpretar automáticamente los clusters en lenguaje de negocio.
6. Generación de dashboards consolidados y reportes exportables.

---

## 🗂️ Estructura del Repositorio
```text
Proyecto SENA/
├── data/
│   ├── raw/
│   │   └── titanic.csv                # Dataset original sin modificar
│   └── processed/
│       ├── titanic_raw_check.csv      # Verificación inicial de ingesta
│       ├── titanic_clean.csv          # Dataset limpio e imputado
│       └── titanic_clustered.csv      # Dataset con asignación de clusters K-Means
├── src/
│   ├── __init__.py                    # Inicializador de módulo
│   ├── 01_ingesta.py                  # Carga y diagnóstico estructural
│   ├── 02_limpieza.py                 # Imputación estadística avanzada
│   ├── 03_eda.py                      # Gráficos y filtros analíticos
│   ├── 04_modelado.py                 # StandardScaler + Método del Codo + K-Means
│   ├── 05_ia_generativa.py            # Consulta a LLM y generación de narrativa
│   └── 06_visualizacion.py            # Dashboard consolidado de analítica
├── outputs/
│   ├── figuras/                       # Gráficos exportados en 150 DPI
│   │   ├── heatmap_correlaciones.png
│   │   ├── boxplot_fare_pclass.png
│   │   ├── countplot_supervivencia.png
│   │   ├── metodo_codo.png
│   │   ├── scatter_clusters.png
│   │   └── dashboard_consolidado.png
│   └── reportes/                      # Reportes de datos y texto
│       ├── resumen_clusters.csv
│       └── interpretacion_ia.txt
├── docs/
│   ├── AA2-EV03_Documento_Maestro.md  # Documento técnico para entrega
│   ├── prompts.md                     # Bitácora de prompts para Antigravity
│   └── guion_video.md                 # Guion minuto a minuto para grabar el video
├── .env.example                       # Variables de entorno de muestra
├── .gitignore                         # Reglas de exclusión para Git
├── requirements.txt                   # Dependencias con versiones fijadas
├── README.md                          # Este documento
└── main.py                            # Orquestador del pipeline completo
```

---

## ⚡ Instalación y Ejecución Rápida

### 1. Clonar o abrir el proyecto en VS Code
Abre la carpeta en Visual Studio Code.

### 2. Crear y activar el entorno virtual
En Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno (Opcional para OpenAI)
Copia el archivo `.env.example` a `.env`:
```powershell
Copy-Item .env.example .env
```
*(Si no configuras una API Key de OpenAI, el pipeline cuenta con un motor analítico generativo integrado de respaldo para que la ejecución no se interrumpa).*

### 5. Ejecutar todo el pipeline
```bash
python main.py
```
O bien ejecutar cada script individualmente:
```bash
python src/01_ingesta.py
python src/02_limpieza.py
python src/03_eda.py
python src/04_modelado.py
python src/05_ia_generativa.py
python src/06_visualizacion.py
```

---

## 📊 Resultados Clave Obtenidos

* **Disparidad de Supervivencia (EDA):**
  * Mujeres de 1ra Clase: **96.81%** de supervivencia ($n=94$).
  * Hombres de 3ra Clase: **13.54%** de supervivencia ($n=347$).
  * Disparidad: Las mujeres en primera clase tuvieron **7.15 veces** mayor probabilidad de rescate.
* **Segmentación K-Means ($k=3$):**
  * **Cluster 0:** 587 pasajeros (Base económica / jóvenes, edad media 21.9 años, tarifa media £19.35, supervivencia 36%).
  * **Cluster 1:** 267 pasajeros (Clase media / adultos, edad media 44.4 años, tarifa media £34.64, supervivencia 40%).
  * **Cluster 2:** 37 pasajeros (Alta gama / ejecutivos, edad media 32.6 años, tarifa media £218.52, supervivencia 73%).

---

## 🎥 Grabación del Video (6 a 8 minutos)
Sigue paso a paso el archivo [`docs/guion_video.md`](file:///d:/WEB/Proyecto%20SENA/docs/guion_video.md) donde encontrarás los tiempos exactos, lo que debes mostrar en pantalla y el texto sugerido para locución, garantizando una calificación perfecta en los 7 criterios de evaluación del SENA.

