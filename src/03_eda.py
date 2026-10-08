"""
Script 03: Análisis Exploratorio de Datos (EDA) y Segmentaciones
Genera visualizaciones analíticas con Seaborn/Matplotlib y realiza filtros/segmentaciones estadísticas.
Evidencia SENA: AA2-EV03
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def configurar_estilo_graficos():
    """Configura el estilo visual homogéneo para las publicaciones."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def generar_heatmap_correlaciones(df: pd.DataFrame, ruta_salida: Path):
    """Genera y guarda el mapa de calor de correlaciones lineales de variables numéricas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cols_numericas = df.select_dtypes(include=["number"]).columns
    corr = df[cols_numericas].corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Matriz de Correlación - Dataset Titanic", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"  📊 Figura guardada: {ruta_salida.name}")


def generar_boxplot_fare_pclass(df: pd.DataFrame, ruta_salida: Path):
    """Genera y guarda un boxplot de la tarifa pagada (Fare) segregada por clase de pasaje (Pclass)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="Pclass",
        y="Fare",
        palette="Set2",
        ax=ax,
        hue="Pclass",
        legend=False
    )
    ax.set_title("Distribución de Tarifas (Fare) por Clase (Pclass)", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Clase de Pasaje (1ra, 2da, 3ra)", fontsize=11)
    ax.set_ylabel("Tarifa Pagada (£)", fontsize=11)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"  📊 Figura guardada: {ruta_salida.name}")


def generar_countplot_supervivencia(df: pd.DataFrame, ruta_salida: Path):
    """Genera y guarda un gráfico de barras comparativo de supervivencia según el sexo."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot = df.copy()
    df_plot["Sexo_Etiqueta"] = df_plot["Sex"].replace({0: "Hombre", 1: "Mujer", "0": "Hombre", "1": "Mujer", "male": "Hombre", "female": "Mujer"})
    df_plot["Supervivencia"] = df_plot["Survived"].replace({0: "No Sobrevivió", 1: "Sobrevivió", "0": "No Sobrevivió", "1": "Sobrevivió"})

    sns.countplot(
        data=df_plot,
        x="Sexo_Etiqueta",
        hue="Supervivencia",
        palette={"No Sobrevivió": "#e74c3c", "Sobrevivió": "#2ecc71"},
        ax=ax
    )
    ax.set_title("Supervivencia de Pasajeros según Género", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Género", fontsize=11)
    ax.set_ylabel("Cantidad de Pasajeros", fontsize=11)
    ax.legend(title="Estado")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"  📊 Figura guardada: {ruta_salida.name}")


def ejecutar_segmentaciones(df: pd.DataFrame) -> dict:
    """
    Aplica filtros y segmentaciones clave respondiendo a preguntas analíticas de negocio.

    Returns:
        dict: Métricas comparativas de las segmentaciones.
    """
    print("\n🔍 --- RESULTADOS DE FILTROS Y SEGMENTACIONES CLAVE ---")
    
    # Segmentación 1: Mujeres de primera clase
    mujeres_1ra = df[((df["Sex"] == 1) | (df["Sex"] == "female")) & (df["Pclass"] == 1)]
    tasa_mujeres_1ra = float(mujeres_1ra["Survived"].mean()) if len(mujeres_1ra) > 0 else 0.0
    total_mujeres_1ra = len(mujeres_1ra)
    print(f"1. Mujeres en 1ra Clase (n={total_mujeres_1ra}): Tasa de Supervivencia = {tasa_mujeres_1ra:.2%}")

    # Segmentación 2: Hombres de tercera clase
    hombres_3ra = df[((df["Sex"] == 0) | (df["Sex"] == "male")) & (df["Pclass"] == 3)]
    tasa_hombres_3ra = float(hombres_3ra["Survived"].mean()) if len(hombres_3ra) > 0 else 0.0
    total_hombres_3ra = len(hombres_3ra)
    print(f"2. Hombres en 3ra Clase (n={total_hombres_3ra}): Tasa de Supervivencia = {tasa_hombres_3ra:.2%}")

    # Segmentación 3: Adultos laboralmente activos con boleto sobre el cuartil 25
    q25_fare = df["Fare"].quantile(0.25)
    filtro_adultos_fare = df[(df["Age"].between(18, 60)) & (df["Fare"] > q25_fare)]
    tasa_adultos_fare = filtro_adultos_fare["Survived"].mean()
    print(f"3. Pasajeros 18-60 años con Tarifa > Q25 (£{q25_fare:.2f}) (n={len(filtro_adultos_fare)}): Supervivencia = {tasa_adultos_fare:.2%}")

    return {
        "mujeres_1ra_tasa": tasa_mujeres_1ra,
        "hombres_3ra_tasa": tasa_hombres_3ra,
        "adultos_fare_tasa": tasa_adultos_fare,
        "ratio_disparidad": (tasa_mujeres_1ra / max(tasa_hombres_3ra, 0.0001))
    }


def ejecutar_eda(ruta_entrada: Path = None, dir_salida: Path = None) -> dict:
    """Orquesta la ejecución de análisis exploratorio y generación de gráficos."""
    base_dir = Path(__file__).resolve().parent.parent
    if ruta_entrada is None:
        ruta_entrada = base_dir / "data" / "processed" / "titanic_clean.csv"
    if dir_salida is None:
        dir_salida = base_dir / "outputs" / "figuras"

    dir_salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ruta_entrada)

    print("📈 Ejecutando Análisis Exploratorio de Datos (EDA)...")
    configurar_estilo_graficos()

    generar_heatmap_correlaciones(df, dir_salida / "heatmap_correlaciones.png")
    generar_boxplot_fare_pclass(df, dir_salida / "boxplot_fare_pclass.png")
    generar_countplot_supervivencia(df, dir_salida / "countplot_supervivencia.png")

    metricas = ejecutar_segmentaciones(df)
    print("✅ EDA completado satisfactoriamente.")
    return metricas


if __name__ == "__main__":
    ejecutar_eda()
