"""
Script 02: Limpieza e Imputación Estadística de Datos
Aplica técnicas estadísticas para tratar valores nulos, transformar variables categóricas
e implementar ingeniería de características sobre el dataset Titanic.
Evidencia SENA: AA2-EV03
"""

from pathlib import Path
import pandas as pd


def limpiar_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y transforma el dataset Titanic aplicando criterios estadísticos rigurosos:

    Justificaciones estadísticas y técnicas:
    1. Eliminación de variables de alta cardinalidad o exceso de nulos:
       - 'Cabin': Posee más del 77% de valores perdidos, por lo que una imputación generaría sesgo artificial.
       - 'Ticket' y 'Name': Cadenas de texto con alta dispersión/cardinalidad no directamente predictivas para clustering sin NLP extenso.
    2. Imputación estadística condicional de 'Age':
       - Se utiliza la mediana agrupada por clase de cabina ('Pclass') y sexo ('Sex').
       - Justificación: La mediana es resistente a valores atípicos (outliers) y la distribución de edad varía significativamente según la clase social y género.
    3. Imputación de 'Embarked' con la moda:
       - Representa la categoría más frecuente (Southampton, 'S'), aplicable a variables categóricas con baja tasa de nulos (2 casos, <0.25%).
    4. Codificación binaria de 'Sex':
       - Mapeo 'male' -> 0, 'female' -> 1 para algoritmos basados en distancia euclidiana.
    5. Ingeniería de características 'FamilySize':
       - Combinación aditiva: SibSp (hermanos/cónyuges) + Parch (padres/hijos) + 1 (el propio pasajero).

    Args:
        df (pd.DataFrame): Dataset crudo original.

    Returns:
        pd.DataFrame: Dataset limpio, imputado y enriquecido.
    """
    df = df.copy()
    nulos_antes = df.isnull().sum().to_dict()
    print("🧹 Iniciando proceso de limpieza estadística...")

    # 1. Eliminación de columnas con exceso de nulos o irrelevantes
    columnas_a_eliminar = ['Cabin', 'Ticket', 'Name']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
    print(f"  [1/5] Columnas eliminadas: {columnas_a_eliminar}")

    # 2. Imputación estadística de 'Age' por mediana condicionada
    if 'Age' in df.columns and 'Pclass' in df.columns and 'Sex' in df.columns:
        mediana_agrupada = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
        df['Age'] = df['Age'].fillna(mediana_agrupada)
        # Respaldo por si quedase algún caso sin grupo
        df['Age'] = df['Age'].fillna(df['Age'].median())
        print("  [2/5] 'Age' imputada con la mediana agrupada por Pclass y Sex.")

    # 3. Imputación de 'Embarked' con la moda estadística
    if 'Embarked' in df.columns:
        moda_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(moda_embarked)
        print(f"  [3/5] 'Embarked' imputada con moda: '{moda_embarked}'.")

    # 4. Codificación numérica de 'Sex'
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
        print("  [4/5] 'Sex' codificado a binario: male=0, female=1.")

    # 5. Ingeniería de características: FamilySize
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        print("  [5/5] Variable 'FamilySize' creada exitosamente.")

    # Resumen de resultados
    nulos_despues = int(df.isnull().sum().sum())
    print(f"✅ Limpieza completada. Total de valores nulos restantes: {nulos_despues}")
    
    return df


def ejecutar_limpieza(ruta_entrada: Path = None, ruta_salida: Path = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline de limpieza y guarda el archivo transformado.

    Args:
        ruta_entrada (Path, opcional): Ruta al dataset crudo.
        ruta_salida (Path, opcional): Ruta al dataset limpio de salida.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if ruta_entrada is None:
        ruta_entrada = base_dir / "data" / "raw" / "titanic.csv"
    if ruta_salida is None:
        ruta_salida = base_dir / "data" / "processed" / "titanic_clean.csv"

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(ruta_entrada)
    df_clean = limpiar_titanic(df_raw)
    df_clean.to_csv(ruta_salida, index=False)
    
    print(f"💾 Dataset limpio exportado a: {ruta_salida}")
    return df_clean


if __name__ == "__main__":
    ejecutar_limpieza()
