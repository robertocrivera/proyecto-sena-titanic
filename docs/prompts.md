# 📝 Bitácora de Prompts para el Agente Generativo (Antigravity)
**Evidencia:** AA2-EV03 - Video Configuración de Analítica de Datos  
**Programa:** Aplicación de la Inteligencia Artificial en la Integración de Datos  
**Resultado de Aprendizaje:** 220501115-02  
**Agente Utilizado:** Antigravity (Google DeepMind / VS Code Assistant)

---

## 📌 Metodología de Intervención con IA
En este proyecto, la IA generativa no reemplaza el criterio técnico del analista de datos, sino que actúa como un copiloto de programación y analítica (pair programmer). Cada fase del flujo CRISP-DM contó con un prompt específico con contexto, restricciones y formato de salida deseado.

---

### Fase 1 & 2: Ingesta de Datos (`src/01_ingesta.py`)
> **Prompt:**
> ```text
> Actúa como un ingeniero de datos. Crea un script modular '01_ingesta.py' en Python que:
> 1. Cargue el archivo CSV 'data/raw/titanic.csv' utilizando pandas.
> 2. Muestre en consola las primeras 5 filas del dataset, la estructura con info() y un resumen ordenado de valores nulos por columna con isnull().sum() junto con sus porcentajes.
> 3. Guarde una copia de respaldo sin transformar en 'data/processed/titanic_raw_check.csv'.
> 4. Incluya docstrings con tipado estricto (type hints) y manejo estructurado de rutas relativas con pathlib.
> ```

---

### Fase 3: Limpieza e Imputación Estadística (`src/02_limpieza.py`)
> **Prompt:**
> ```text
> Actúa como un científico de datos especializado en preprocesamiento. Crea el script '02_limpieza.py' con una función 'limpiar_titanic(df)' que implemente las siguientes transformaciones justificadas estadísticamente:
> 1. Elimine las columnas 'Cabin', 'Ticket' y 'Name' (justificando el descarte por alta cardinalidad y más del 77% de nulos).
> 2. Impute los valores nulos de 'Age' utilizando la mediana agrupada por 'Pclass' y 'Sex' (evitando sesgos globales de la mediana simple).
> 3. Impute los nulos de 'Embarked' utilizando la moda estadística.
> 4. Codifique 'Sex' a una representación binaria (0 = male, 1 = female).
> 5. Aplique ingeniería de características creando 'FamilySize' = SibSp + Parch + 1.
> 6. Exporte el dataset limpio a 'data/processed/titanic_clean.csv'.
> Añade comentarios detallados y docstrings justificando cada decisión técnica.
> ```

---

### Fase 4: Análisis Exploratorio de Datos (EDA) y Segmentaciones (`src/03_eda.py`)
> **Prompt:**
> ```text
> Crea un script '03_eda.py' para análisis exploratorio que genere y guarde visualizaciones analíticas de alta calidad (150 dpi) en 'outputs/figuras/':
> 1. 'heatmap_correlaciones.png': Mapa de calor con anotaciones numéricas de correlación lineal entre variables numéricas.
> 2. 'boxplot_fare_pclass.png': Diagrama de caja comparando la distribución de tarifas por clase de pasaje.
> 3. 'countplot_supervivencia.png': Gráfico de barras comparando supervivencia según género.
> 4. Implementa además filtros y segmentaciones clave respondiendo a hipótesis de negocio:
>    - Supervivencia en mujeres de 1ra clase vs hombres de 3ra clase.
>    - Pasajeros entre 18 y 60 años con tarifa superior al percentil 25.
> Muestra los porcentajes y métricas clave en consola.
> ```

---

### Fase 5: Modelado no Supervisado con K-Means (`src/04_modelado.py`)
> **Prompt:**
> ```text
> Desarrolla el script '04_modelado.py' para segmentación de pasajeros utilizando scikit-learn:
> 1. Escale las variables 'Age' y 'Fare' utilizando StandardScaler (explicando por qué es mandatorio para algoritmos basados en distancias euclidianas).
> 2. Implemente el método del codo evaluando k de 2 a 10 con la métrica de inercia WCSS y guarde el gráfico en 'outputs/figuras/metodo_codo.png'.
> 3. Entrene el modelo KMeans definitivo con k=3 clusters.
> 4. Asigne la columna 'cluster' al dataset y genere un scatter plot (Age vs Fare) guardado en 'outputs/figuras/scatter_clusters.png'.
> 5. Calcule el resumen estadístico descriptivo de cada cluster (conteo, medias, medianas y tasa de supervivencia) y expórtelo a 'outputs/reportes/resumen_clusters.csv'.
> ```

---

### Fase 6: Integración de IA Generativa en el Pipeline (`src/05_ia_generativa.py`)
> **Prompt:**
> ```text
> Crea un módulo '05_ia_generativa.py' que consuma el resumen de clusters ('outputs/reportes/resumen_clusters.csv') y:
> 1. Utilice el SDK oficial de OpenAI ('openai>=1.30.0') con el modelo 'gpt-4o-mini', leyendo la API Key de forma segura desde un archivo '.env'.
> 2. Diseñe un system prompt y user prompt especializado para que el LLM traduzca las métricas numéricas a un informe de negocio formal (perfiles de pasajeros, impacto económico y conclusiones operativas).
> 3. Incluya un mecanismo inteligente de fallback offline para que el pipeline no falle en caso de ausencia temporal de credenciales.
> 4. Guarde el informe generado en 'outputs/reportes/interpretacion_ia.txt'.
> ```

---

### Fase 7: Orquestación del Pipeline (`main.py`)
> **Prompt:**
> ```text
> Crea un script orquestador 'main.py' que ejecute de forma secuencial y modular cada una de las fases del pipeline (ingesta, limpieza, eda, modelado, ia generativa y dashboard de visualización).
> Debe incluir banners de estado en consola, medición del tiempo de ejecución total y manejo de excepciones con trazabilidad de errores.
> Carga los módulos dinámicamente mediante 'importlib' para soportar los nombres de archivo estandarizados del SENA.
> ```

