"""
Script 04: Modelado con K-Means Clustering y Método del Codo
Aplica escalado estándar, determina el número óptimo de clusters k, segmenta a los pasajeros
y exporta gráficos y reportes estadísticos.
Evidencia SENA: AA2-EV03
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calcular_metodo_codo(X_scaled, k_range=range(2, 11)) -> list[float]:
    """
    Calcula la inercia (WCSS: Within-Cluster Sum of Squares) para un rango de valores de k.

    Args:
        X_scaled (np.ndarray): Matriz de datos normalizados.
        k_range (range): Rango de k a evaluar (de 2 a 10).

    Returns:
        list[float]: Valores de inercia para cada k.
    """
    inercias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
    return inercias


def graficar_metodo_codo(k_range, inercias, ruta_salida: Path):
    """Genera y guarda el gráfico del método del codo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inercias, marker='o', linewidth=2, color="#2980b9", markersize=8)
    ax.set_title("Método del Codo para Selección de k Óptimo (K-Means)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de Clusters (k)", fontsize=11)
    ax.set_ylabel("Inercia (WCSS)", fontsize=11)
    ax.set_xticks(list(k_range))
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Destacar k=3 como punto de inflexión
    ax.axvline(x=3, color='#e74c3c', linestyle=':', label='k óptimo sugerido (k=3)')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"  📊 Gráfico del codo guardado en: {ruta_salida.name}")


def graficar_scatter_clusters(df: pd.DataFrame, ruta_salida: Path):
    """Genera un scatter plot Age vs Fare coloreado por el cluster asignado."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Fare",
        hue="cluster",
        palette="viridis",
        style="cluster",
        s=70,
        alpha=0.85,
        ax=ax
    )
    ax.set_title("Segmentación de Pasajeros: K-Means (Age vs Fare)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Edad (Años)", fontsize=11)
    ax.set_ylabel("Tarifa Pagada (£)", fontsize=11)
    ax.legend(title="Cluster Asignado", loc="upper right")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"  📊 Gráfico de dispersión de clusters guardado en: {ruta_salida.name}")


def ejecutar_modelado(ruta_entrada: Path = None, dir_figuras: Path = None, dir_reportes: Path = None) -> pd.DataFrame:
    """
    Orquesta el pipeline completo de machine learning no supervisado.

    Returns:
        pd.DataFrame: Resumen estadístico de los clusters formados.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if ruta_entrada is None:
        ruta_entrada = base_dir / "data" / "processed" / "titanic_clean.csv"
    if dir_figuras is None:
        dir_figuras = base_dir / "outputs" / "figuras"
    if dir_reportes is None:
        dir_reportes = base_dir / "outputs" / "reportes"

    dir_figuras.mkdir(parents=True, exist_ok=True)
    dir_reportes.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ruta_entrada)
    print("🤖 Iniciando proceso de modelado no supervisado (K-Means)...")

    # 1. Selección y escalado de características
    # Es mandatorio en algoritmos basados en distancias (como K-Means) para evitar que 'Fare' domine a 'Age'
    X = df[['Age', 'Fare']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Evaluación del número óptimo de clusters (Método del Codo)
    k_range = range(2, 11)
    inercias = calcular_metodo_codo(X_scaled, k_range)
    graficar_metodo_codo(k_range, inercias, dir_figuras / "metodo_codo.png")

    # 3. Entrenamiento con k=3
    k_optimo = 3
    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Gráfico de dispersión de clusters
    graficar_scatter_clusters(df, dir_figuras / "scatter_clusters.png")

    # 5. Generación de resumen estadístico por cluster
    resumen = df.groupby('cluster').agg(
        total_pasajeros=('PassengerId', 'count'),
        edad_media=('Age', 'mean'),
        edad_mediana=('Age', 'median'),
        tarifa_media=('Fare', 'mean'),
        tarifa_mediana=('Fare', 'median'),
        tasa_supervivencia=('Survived', 'mean')
    ).round(2)

    ruta_resumen_csv = dir_reportes / "resumen_clusters.csv"
    resumen.to_csv(ruta_resumen_csv)
    print(f"💾 Resumen de clusters exportado a: {ruta_resumen_csv}")
    print("\n--- RESUMEN ESTADÍSTICO DE CLUSTERS (K-Means k=3) ---")
    print(resumen)

    # Guardar también el dataset etiquetado con clusters
    ruta_df_clustered = base_dir / "data" / "processed" / "titanic_clustered.csv"
    df.to_csv(ruta_df_clustered, index=False)

    return resumen


if __name__ == "__main__":
    ejecutar_modelado()

