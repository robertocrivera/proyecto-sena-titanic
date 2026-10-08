"""
Script 06: Visualización Consolidada y Dashboard de Analítica
Genera un panel gráfico consolidado (dashboard estático) con las principales métricas
del análisis exploratorio y del clustering de K-Means.
Evidencia SENA: AA2-EV03
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generar_dashboard_analitico(ruta_datos: Path, dir_figuras: Path):
    """
    Crea un panel visual consolidado 2x2 que reúne:
    1. Distribución de Edad por Supervivencia
    2. Supervivencia por Clase y Género
    3. Distribución de Tarifas por Cluster K-Means
    4. Proporción de Pasajeros por Cluster
    """
    df = pd.read_csv(ruta_datos)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dashboard Consolidado de Analítica de Datos - Titanic (SENA AA2-EV03)", fontsize=16, fontweight="bold", y=0.98)

    # 1. KDE Plot: Edad vs Supervivencia
    ax1 = axes[0, 0]
    sns.kdeplot(data=df[df['Survived'] == 1], x='Age', label='Sobrevivió', fill=True, color='#2ecc71', ax=ax1, alpha=0.4)
    sns.kdeplot(data=df[df['Survived'] == 0], x='Age', label='No Sobrevivió', fill=True, color='#e74c3c', ax=ax1, alpha=0.4)
    ax1.set_title("1. Distribución de Densidad de Edad por Supervivencia", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Edad (Años)")
    ax1.set_ylabel("Densidad")
    ax1.legend()

    # 2. Barplot: Tasa de Supervivencia por Clase y Género
    ax2 = axes[0, 1]
    df_plot2 = df.copy()
    df_plot2['Sexo'] = df_plot2['Sex'].replace({0: 'Hombre', 1: 'Mujer', '0': 'Hombre', '1': 'Mujer', 'male': 'Hombre', 'female': 'Mujer'})
    sns.barplot(data=df_plot2, x='Pclass', y='Survived', hue='Sexo', palette={'Hombre': '#3498db', 'Mujer': '#e84393'}, ax=ax2, errorbar=None)
    ax2.set_title("2. Tasa de Supervivencia por Clase y Género", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Clase (Pclass)")
    ax2.set_ylabel("Tasa de Supervivencia")
    ax2.set_ylim(0, 1)

    # 3. Boxplot: Tarifa por Cluster K-Means
    ax3 = axes[1, 0]
    if 'cluster' in df.columns:
        sns.boxplot(data=df, x='cluster', y='Fare', palette='viridis', ax=ax3, hue='cluster', legend=False)
        ax3.set_title("3. Distribución de Tarifas por Cluster K-Means", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Cluster K-Means")
        ax3.set_ylabel("Tarifa (£)")
    else:
        ax3.text(0.5, 0.5, "Requiere columna 'cluster'", ha='center')

    # 4. Conteo y Tasa de Supervivencia por Cluster
    ax4 = axes[1, 1]
    if 'cluster' in df.columns:
        resumen_cluster = df.groupby('cluster')['Survived'].mean().reset_index()
        bars = ax4.bar(resumen_cluster['cluster'].astype(str), resumen_cluster['Survived'], color=['#3498db', '#9b59b6', '#e67e22'])
        ax4.set_title("4. Tasa de Supervivencia por Cluster Identificado", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Cluster K-Means")
        ax4.set_ylabel("Tasa de Supervivencia")
        ax4.set_ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1%}", ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "Requiere columna 'cluster'", ha='center')

    plt.tight_layout()
    ruta_salida = dir_figuras / "dashboard_consolidado.png"
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"📊 Dashboard consolidado generado exitosamente en: {ruta_salida}")


def ejecutar_visualizaciones():
    """Ejecuta la generación de visualizaciones finales."""
    base_dir = Path(__file__).resolve().parent.parent
    ruta_datos = base_dir / "data" / "processed" / "titanic_clustered.csv"
    if not ruta_datos.exists():
        ruta_datos = base_dir / "data" / "processed" / "titanic_clean.csv"
    dir_figuras = base_dir / "outputs" / "figuras"
    dir_figuras.mkdir(parents=True, exist_ok=True)

    generar_dashboard_analitico(ruta_datos, dir_figuras)


if __name__ == "__main__":
    ejecutar_visualizaciones()
