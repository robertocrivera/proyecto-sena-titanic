"""
Aplicación Web Interactiva - Dashboard de Analítica de Datos con IA Generativa
Evidencia SENA: AA2-EV03
Programa: Aplicación de la Inteligencia Artificial en la Integración de Datos
Resultado de Aprendizaje: 220501115-02
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="Analítica de Datos & IA - SENA AA2-EV03",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Evitar que traductores automáticos de navegadores (como Chrome Translate) 
# alteren nodos del DOM de React y causen el error 'removeChild'
st.markdown("""
<meta name="google" content="notranslate">
<style>
    .stApp {
        translate: no !important;
    }
</style>
""", unsafe_allow_html=True)

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DIR_FIGURAS = BASE_DIR / "outputs" / "figuras"
DIR_REPORTES = BASE_DIR / "outputs" / "reportes"
DIR_DATA = BASE_DIR / "data" / "processed"
LOGO_SENA = BASE_DIR / "web" / "sena_logo_verde.svg"


@st.cache_data
def cargar_datos():
    ruta_clustered = DIR_DATA / "titanic_clustered.csv"
    ruta_clean = DIR_DATA / "titanic_clean.csv"
    if ruta_clustered.exists():
        df = pd.read_csv(ruta_clustered)
    elif ruta_clean.exists():
        df = pd.read_csv(ruta_clean)
    else:
        df = None
    return df


@st.cache_data
def cargar_reportes():
    ruta_resumen = DIR_REPORTES / "resumen_clusters.csv"
    ruta_ia = DIR_REPORTES / "interpretacion_ia.txt"

    df_resumen = pd.read_csv(ruta_resumen) if ruta_resumen.exists() else None
    texto_ia = ruta_ia.read_text(encoding="utf-8") if ruta_ia.exists() else "Reporte no encontrado."
    return df_resumen, texto_ia


# Carga de datos
df = cargar_datos()
df_resumen, texto_ia = cargar_reportes()

# --- SIDEBAR ---
with st.sidebar:
    if LOGO_SENA.exists():
        st.image(str(LOGO_SENA), width=100)
    else:
        st.markdown("## 🟢 SENA")
        
    st.title("SENA | Analítica & IA")
    st.caption("**Evidencia:** AA2-EV03\n\n**RA:** 220501115-02\n\n**Agente:** Antigravity (IA Generativa)")
    st.markdown("---")

    menu = st.radio(
        "Navegación:",
        [
            "🏛️ Resumen Ejecutivo & KPIs",
            "📈 Análisis Exploratorio (EDA)",
            "🤖 Modelado Machine Learning (K-Means)",
            "🧠 Interpretación con IA Generativa",
            "📋 Documentación & Guion de Video"
        ]
    )

    st.markdown("---")
    st.info("💡 **Dataset:** Titanic (891 registros)\n\n**Pipeline:** Python + Pandas + Scikit-Learn + OpenAI")

# --- HEADER PRINCIPAL NATIVO (100% React Seguro) ---
with st.container(border=True):
    st.title("🚢 Plataforma de Analítica de Datos & Inteligencia Artificial", anchor=False)
    st.caption("Visualización interactiva de métricas estadísticas, modelos de clustering K-Means e interpretación automatizada con LLM.")

# ==============================================================================
# SECCIÓN 1: RESUMEN EJECUTIVO & KPIS
# ==============================================================================
if menu == "🏛️ Resumen Ejecutivo & KPIs":
    st.subheader("📌 Indicadores Clave de Desempeño (KPIs)")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        total_pasajeros = len(df)
        tasa_supervivencia = df["Survived"].mean()
        
        # Segmentaciones
        mujeres_1ra = df[((df["Sex"] == 1) | (df["Sex"] == "female")) & (df["Pclass"] == 1)]
        hombres_3ra = df[((df["Sex"] == 0) | (df["Sex"] == "male")) & (df["Pclass"] == 3)]
        
        tasa_m1 = mujeres_1ra["Survived"].mean() if len(mujeres_1ra) > 0 else 0
        tasa_h3 = hombres_3ra["Survived"].mean() if len(hombres_3ra) > 0 else 0

        with col1:
            with st.container(border=True):
                st.caption("TOTAL PASAJEROS")
                st.markdown(f"### {total_pasajeros:,}")
                st.caption("100% de la muestra")

        with col2:
            with st.container(border=True):
                st.caption("SUPERVIVENCIA GLOBAL")
                st.markdown(f"### {tasa_supervivencia:.1%}")
                st.caption("342 sobrevivientes")

        with col3:
            with st.container(border=True):
                st.caption("MUJERES 1RA CLASE")
                st.markdown(f"### :green[{tasa_m1:.1%}]")
                st.caption(f"{len(mujeres_1ra)} pasajeras")

        with col4:
            with st.container(border=True):
                st.caption("HOMBRES 3RA CLASE")
                st.markdown(f"### :red[{tasa_h3:.1%}]")
                st.caption(f"{len(hombres_3ra)} pasajeros")

    st.markdown("---")
    st.subheader("📊 Dashboard Consolidado de Analítica de Datos")
    
    ruta_dashboard = DIR_FIGURAS / "dashboard_consolidado.png"
    if ruta_dashboard.exists():
        img_dash = Image.open(ruta_dashboard)
        st.image(img_dash, caption="Panel integral 2x2: Densidad de edad, disparidad de género/clase y comportamiento de tarifas.", width="stretch")
    else:
        st.warning("El dashboard consolidado no ha sido generado aún. Ejecuta `python main.py`.")

    with st.expander("📖 Lectura e Interpretación del Dashboard Consolidado"):
        st.markdown("""
        * **1. Densidad de Edad vs Supervivencia:** Se observa un pico de supervivencia en niños menores de 10 años, confirmando la aplicación del protocolo *"mujeres y niños primero"*. Los adultos entre 20 y 35 años sufrieron la mayor tasa de mortalidad.
        * **2. Tasa por Clase y Género:** La brecha de género es contundente; las mujeres de primera y segunda clase superan el 90% de supervivencia, mientras que los hombres en tercera clase caen al 13.5%.
        * **3. Tarifas por Cluster:** El Cluster 2 agrupa exclusivamente a la élite económica, con tarifas atípicas sobre £200.
        * **4. Supervivencia por Cluster:** El factor económico se correlaciona fuertemente con la supervivencia, ascendiendo de 36% en el cluster popular al 73% en el cluster premium.
        """)

# ==============================================================================
# SECCIÓN 2: ANÁLISIS EXPLORATORIO (EDA)
# ==============================================================================
elif menu == "📈 Análisis Exploratorio (EDA)":
    st.subheader("🔍 Análisis Exploratorio de Datos y Segmentaciones")
    st.write("Explora las visualizaciones estadísticas generadas con Seaborn y Matplotlib con su respectiva justificación analítica:")

    tab1, tab2, tab3 = st.tabs([
        "🔥 Matriz de Correlación",
        "📦 Distribución de Tarifas (Boxplot)",
        "👥 Supervivencia por Género (Countplot)"
    ])

    with tab1:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ruta_img = DIR_FIGURAS / "heatmap_correlaciones.png"
            if ruta_img.exists():
                st.image(Image.open(ruta_img), width="stretch")
        with c2:
            st.markdown("### 📊 Lectura de la Matriz de Correlación")
            st.markdown("""
            * **Correlación `Survived` vs `Sex` (+0.54):** Es la correlación lineal más fuerte de todo el conjunto. Refleja la prioridad estricta de género en el protocolo de evacuación.
            * **Correlación `Survived` vs `Pclass` (-0.34):** Correlación negativa significativa: a mayor número de clase (3ra clase), menor probabilidad de supervivencia.
            * **Correlación `Fare` vs `Pclass` (-0.55):** Consistente con la lógica de negocio; pasajes de 1ra clase tenían precios sustancialmente mayores.
            * **Correlación `FamilySize` vs `SibSp`/`Parch` (>+0.75):** Valida la ingeniería de características realizada en la etapa de limpieza.
            """)

    with tab2:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ruta_img = DIR_FIGURAS / "boxplot_fare_pclass.png"
            if ruta_img.exists():
                st.image(Image.open(ruta_img), width="stretch")
        with c2:
            st.markdown("### 📦 Lectura del Boxplot (Fare por Pclass)")
            st.markdown("""
            * **1ra Clase:** Alta dispersión con valores atípicos (*outliers*) superiores a £200 y hasta £512 (suites presidenciales). Mediana cercana a £60.
            * **2da Clase:** Distribución homogénea y compacta con mediana alrededor de £14.
            * **3ra Clase:** Tarifas muy bajas, altamente concentradas entre £7 y £15.
            * **Conclusión Técnica:** La presencia de valores atípicos masivos en `Fare` hace **obligatorio** el escalado con `StandardScaler` previo al modelado con K-Means.
            """)

    with tab3:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ruta_img = DIR_FIGURAS / "countplot_supervivencia.png"
            if ruta_img.exists():
                st.image(Image.open(ruta_img), width="stretch")
        with c2:
            st.markdown("### 👥 Lectura del Conteo de Supervivencia")
            st.markdown("""
            * **Población Femenina:** Más del **74%** de las mujeres a bordo lograron salvarse.
            * **Población Masculina:** Menos del **19%** de los hombres sobrevivieron; el 81% pereció.
            * **Hipótesis Validada:** La orden de embarque en botes salvavidas fue el filtro más determinante en el desenlace del siniestro.
            """)

# ==============================================================================
# SECCIÓN 3: MODELADO MACHINE LEARNING (K-MEANS)
# ==============================================================================
elif menu == "🤖 Modelado Machine Learning (K-Means)":
    st.subheader("🤖 Segmentación no Supervisada: K-Means Clustering")
    st.write("Implementación del algoritmo de agrupamiento sobre las variables demográfica (`Age`) y económica (`Fare`).")

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("#### 1. Método del Codo (Selección de $k$)")
        ruta_codo = DIR_FIGURAS / "metodo_codo.png"
        if ruta_codo.exists():
            st.image(Image.open(ruta_codo), width="stretch")
        st.caption("Evolución de la inercia WCSS (Within-Cluster Sum of Squares) para $k \\in [2, 10]$.")
        st.info("📌 **Justificación:** El quiebre más notorio de la curva ocurre en **$k=3$**, logrando una excelente cohesión interna sin sobrefragmentar la población.")

    with col_der:
        st.markdown("#### 2. Dispersión de Clusters (Age vs Fare)")
        ruta_scatter = DIR_FIGURAS / "scatter_clusters.png"
        if ruta_scatter.exists():
            st.image(Image.open(ruta_scatter), width="stretch")
        st.caption("Partición del espacio euclidiano normalizado proyectado sobre variables originales.")
        st.success("📌 **Resultado:** Separación nítida entre la base económica, la clase media adulta y los pasajeros de alta gama.")

    st.markdown("---")
    st.subheader("📋 Resumen Cuantitativo por Cluster")
    if df_resumen is not None:
        st.dataframe(df_resumen, width="stretch")
    else:
        st.warning("No se encontró `resumen_clusters.csv`.")

# ==============================================================================
# SECCIÓN 4: IA GENERATIVA INTEGRADA
# ==============================================================================
elif menu == "🧠 Interpretación con IA Generativa":
    st.subheader("🧠 Integración de Modelos de Lenguaje (LLM) en el Pipeline")
    st.write("La IA Generativa actúa como consultor de analítica de negocio, interpretando de forma autónoma los datos numéricos.")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("### 📄 Informe Generado por la IA")
        st.text_area("Contenido de outputs/reportes/interpretacion_ia.txt", texto_ia, height=450)

    with col2:
        st.markdown("### 🎯 Perfiles de Pasajeros Sintetizados")
        
        with st.container(border=True):
            st.markdown("**🟢 Cluster 0: Pasajeros Económicos y Jóvenes**")
            st.markdown("• **587 personas** (66% del total)\n\n• Tarifa mediana: **£10.50** | Edad media: **~22 años**\n\n• Supervivencia: :red[**36.1%**] (baja)")

        with st.container(border=True):
            st.markdown("**🟠 Cluster 1: Adultos y Clase Media**")
            st.markdown("• **267 personas** (30% del total)\n\n• Tarifa mediana: **£26.39** | Edad media: **~44 años**\n\n• Supervivencia: :orange[**40.1%**] (moderada)")

        with st.container(border=True):
            st.markdown("**🔵 Cluster 2: Élite / Pasajeros de Alta Gama**")
            st.markdown("• **37 personas** (4% del total)\n\n• Tarifa mediana: **£211.34** | Edad media: **~33 años**\n\n• Supervivencia: :green[**73.0%**] (muy alta)")

        st.info("💡 **Diferenciador Técnico:** El pipeline demuestra cómo un analista puede encadenar modelos de ML con LLMs para automatizar la reportería directiva.")

# ==============================================================================
# SECCIÓN 5: DOCUMENTACIÓN & GUION DE VIDEO
# ==============================================================================
elif menu == "📋 Documentación & Guion de Video":
    st.subheader("📋 Documentación de Respaldo y Criterios de Evaluación")

    tab_eval, tab_guion, tab_stack = st.tabs([
        "✅ Criterios de Evaluación SENA",
        "🎬 Guion Minuto a Minuto del Video",
        "🛠️ Stack Tecnológico & Arquitectura"
    ])

    with tab_eval:
        st.markdown("""
        | # | Criterio de Evaluación | Estado en el Proyecto | Minuto en Video |
        | :-: | :--- | :---: | :---: |
        | **1** | **Paso a paso de configuración inicial** | ✅ Cumplido (`venv`, `requirements.txt`, VS Code) | `01:45 - 02:45` |
        | **2** | **Justificación de la herramienta** | ✅ Cumplido (Python vs Power BI / Tableau) | `00:45 - 01:45` |
        | **3** | **Importación y limpieza** | ✅ Cumplido (Mediana agrupada y moda) | `02:45 - 04:00` |
        | **4** | **Filtros y segmentaciones** | ✅ Cumplido (Mujeres 1ra clase 96.8% vs Hombres 3ra clase 13.5%) | `04:00 - 05:00` |
        | **5** | **Algoritmo de analítica** | ✅ Cumplido (StandardScaler + Método del codo + K-Means) | `05:00 - 06:15` |
        | **6** | **Narración clara y técnica** | ✅ Cumplido (Guion ensayado en `docs/guion_video.md`) | `Todo el video` |
        | **7** | **Documento PDF con datos y enlace** | ✅ Cumplido (`docs/AA2-EV03_Documento_Maestro.md`) | `Entrega final` |
        """)

    with tab_guion:
        ruta_guion = BASE_DIR / "docs" / "guion_video.md"
        if ruta_guion.exists():
            st.markdown(ruta_guion.read_text(encoding="utf-8"))
        else:
            st.warning("No se encontró el archivo del guion.")

    with tab_stack:
        st.markdown("""
        * **Lenguaje:** Python 3.12 (64-bit)
        * **Framework Web:** Streamlit 1.62.0
        * **Manipulación de Datos:** Pandas 3.0 & NumPy 2.4
        * **Visualización:** Seaborn 0.13, Matplotlib 3.11
        * **Machine Learning:** Scikit-Learn 1.9 (KMeans, StandardScaler)
        * **IA Generativa:** OpenAI SDK (`gpt-4o-mini`) + Fallback analítico
        * **Agente Generativo:** Antigravity (VS Code Assistant)
        """)

# --- FOOTER NATIVO ---
st.markdown("---")
st.caption("SENA - Servicio Nacional de Aprendizaje | Programa: Aplicación de la Inteligencia Artificial en la Integración de Datos | 2026")
