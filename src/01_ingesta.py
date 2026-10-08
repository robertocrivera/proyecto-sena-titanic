"""
Script 01: Ingesta de datos
Carga el dataset Titanic desde data/raw/ y genera un reporte inicial de estructura y valores nulos.
Evidencia SENA: AA2-EV03
"""

from pathlib import Path
import pandas as pd


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Args:
        ruta (str | Path): Ruta al archivo de datos crudos.

    Returns:
        pd.DataFrame: DataFrame cargado con la información original.
    """
    df = pd.read_csv(ruta)
    print(f"✅ Dataset cargado exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df


def reporte_inicial(df: pd.DataFrame) -> dict:
    """
    Genera y muestra en consola un reporte de la estructura, tipos de datos y valores nulos.

    Args:
        df (pd.DataFrame): DataFrame a auditar.

    Returns:
        dict: Resumen estadístico de dimensiones y conteo de nulos.
    """
    print("\n" + "=" * 50)
    print("📋 REPORTE INICIAL DE INGESTA DE DATOS")
    print("=" * 50)
    
    print("\n--- Primeras 5 filas ---")
    print(df.head())
    
    print("\n--- Información general del DataFrame ---")
    df.info()
    
    print("\n--- Conteo de valores nulos por columna ---")
    nulos = df.isnull().sum()
    nulos_tabla = pd.DataFrame({
        "Nulos": nulos,
        "Porcentaje (%)": (nulos / len(df) * 100).round(2)
    }).sort_values(by="Nulos", ascending=False)
    print(nulos_tabla)
    
    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "total_nulos": int(df.isnull().sum().sum()),
        "columnas_con_nulos": nulos[nulos > 0].to_dict()
    }


def ejecutar_ingesta(ruta_entrada: Path = None, ruta_salida: Path = None) -> pd.DataFrame:
    """
    Función orquestadora para la etapa de ingesta de datos.

    Args:
        ruta_entrada (Path, opcional): Ruta al CSV crudo.
        ruta_salida (Path, opcional): Ruta de respaldo procesado.

    Returns:
        pd.DataFrame: DataFrame verificado.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if ruta_entrada is None:
        ruta_entrada = base_dir / "data" / "raw" / "titanic.csv"
    if ruta_salida is None:
        ruta_salida = base_dir / "data" / "processed" / "titanic_raw_check.csv"

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(ruta_entrada)
    reporte_inicial(df)
    
    df.to_csv(ruta_salida, index=False)
    print(f"\n💾 Respaldo de verificación guardado en: {ruta_salida}")
    return df


if __name__ == "__main__":
    ejecutar_ingesta()

