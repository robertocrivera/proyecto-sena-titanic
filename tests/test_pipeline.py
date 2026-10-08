"""
Suite de Pruebas Automatizadas y Verificación de Seguridad
Proyecto: AA2-EV03 - Analítica de Datos & IA Generativa
SENA - 2026
"""

import unittest
from pathlib import Path
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent


class TestIntegridadDatos(unittest.TestCase):
    """Pruebas unitarias de integridad de datos crudos y procesados."""

    def test_01_dataset_crudo_existe_y_valido(self):
        ruta_raw = BASE_DIR / "data" / "raw" / "titanic.csv"
        self.assertTrue(ruta_raw.exists(), "El dataset crudo titanic.csv debe existir.")
        df_raw = pd.read_csv(ruta_raw)
        self.assertEqual(df_raw.shape[0], 891, "El dataset crudo debe tener exactamente 891 filas.")
        self.assertGreaterEqual(df_raw.shape[1], 12, "El dataset crudo debe tener 12 columnas.")

    def test_02_dataset_limpio_cero_nulos(self):
        ruta_clean = BASE_DIR / "data" / "processed" / "titanic_clean.csv"
        self.assertTrue(ruta_clean.exists(), "El dataset titanic_clean.csv debe existir.")
        df_clean = pd.read_csv(ruta_clean)
        total_nulos = int(df_clean.isnull().sum().sum())
        self.assertEqual(total_nulos, 0, f"El dataset limpio debe tener 0 nulos, encontrados: {total_nulos}.")

    def test_03_codificacion_binaria_sexo(self):
        ruta_clean = BASE_DIR / "data" / "processed" / "titanic_clean.csv"
        df_clean = pd.read_csv(ruta_clean)
        valores_sex = set(df_clean["Sex"].unique())
        self.assertTrue(valores_sex.issubset({0, 1}), f"La columna Sex debe ser binaria (0, 1). Valores: {valores_sex}")

    def test_04_ingenieria_caracteristicas_family_size(self):
        ruta_clean = BASE_DIR / "data" / "processed" / "titanic_clean.csv"
        df_clean = pd.read_csv(ruta_clean)
        self.assertIn("FamilySize", df_clean.columns, "La columna FamilySize debe haber sido creada.")
        esperado = df_clean["SibSp"] + df_clean["Parch"] + 1
        diferencias = (df_clean["FamilySize"] != esperado).sum()
        self.assertEqual(diferencias, 0, "FamilySize debe ser exactamente igual a SibSp + Parch + 1.")


class TestModeladoYArtefactos(unittest.TestCase):
    """Pruebas de modelado de Machine Learning y artefactos generados."""

    def test_05_clusters_generados_correctamente(self):
        ruta_clustered = BASE_DIR / "data" / "processed" / "titanic_clustered.csv"
        self.assertTrue(ruta_clustered.exists(), "El dataset titanic_clustered.csv debe existir.")
        df_clustered = pd.read_csv(ruta_clustered)
        self.assertIn("cluster", df_clustered.columns, "Debe existir la columna 'cluster'.")
        clusters_unicos = set(df_clustered["cluster"].unique())
        self.assertEqual(clusters_unicos, {0, 1, 2}, f"Deben existir exactamente 3 clusters (0, 1, 2). Encontrados: {clusters_unicos}")

    def test_06_resumen_clusters_csv(self):
        ruta_resumen = BASE_DIR / "outputs" / "reportes" / "resumen_clusters.csv"
        self.assertTrue(ruta_resumen.exists(), "El archivo resumen_clusters.csv debe existir.")
        df_resumen = pd.read_csv(ruta_resumen)
        self.assertEqual(len(df_resumen), 3, "El resumen debe reportar exactamente 3 clusters.")

    def test_07_informe_ia_generado(self):
        ruta_ia = BASE_DIR / "outputs" / "reportes" / "interpretacion_ia.txt"
        self.assertTrue(ruta_ia.exists(), "El archivo interpretacion_ia.txt debe existir.")
        contenido = ruta_ia.read_text(encoding="utf-8")
        self.assertGreater(len(contenido), 200, "El informe de IA debe contener texto sustantivo.")

    def test_08_todas_las_figuras_existen_y_no_estan_corruptas(self):
        figuras_esperadas = [
            "heatmap_correlaciones.png",
            "boxplot_fare_pclass.png",
            "countplot_supervivencia.png",
            "metodo_codo.png",
            "scatter_clusters.png",
            "dashboard_consolidado.png"
        ]
        dir_figuras = BASE_DIR / "outputs" / "figuras"
        for nombre in figuras_esperadas:
            ruta = dir_figuras / nombre
            self.assertTrue(ruta.exists(), f"La figura {nombre} debe existir.")
            self.assertGreater(ruta.stat().st_size, 5000, f"La figura {nombre} no debe estar vacía.")
            # Verificar integridad de imagen
            with Image.open(ruta) as img:
                img.verify()


class TestSeguridadYAuditoria(unittest.TestCase):
    """Pruebas de seguridad defensiva, pentesting estático y buenas prácticas."""

    def test_09_sin_claves_api_hardcoded(self):
        """Verifica que ninguna API Key real de OpenAI (sk-...) esté expuesta en el código."""
        archivos_py = list((BASE_DIR / "src").glob("*.py")) + [BASE_DIR / "main.py", BASE_DIR / "app.py"]
        for archivo in archivos_py:
            contenido = archivo.read_text(encoding="utf-8")
            self.assertNotIn("sk-proj-", contenido, f"Vulnerabilidad de fuga de credencial detectada en {archivo.name}.")
            # Verificar que no haya sk- de más de 20 caracteres que no sea placeholder
            lineas = contenido.splitlines()
            for nro, linea in enumerate(lineas, 1):
                if "sk-" in linea and "placeholder" not in linea.lower():
                    # Si no es un comentario o comprobación de fallback
                    partes = [p for p in linea.split() if p.startswith("sk-") and len(p) > 25]
                    self.assertEqual(len(partes), 0, f"Posible API Key expuesta en {archivo.name}:{nro}")

    def test_10_gitignore_protege_secretos_y_entornos(self):
        """Verifica que .gitignore excluya .env y entornos virtuales."""
        ruta_gitignore = BASE_DIR / ".gitignore"
        self.assertTrue(ruta_gitignore.exists(), ".gitignore debe existir.")
        contenido = ruta_gitignore.read_text(encoding="utf-8")
        self.assertIn(".env", contenido, ".env debe estar en .gitignore para prevenir filtración.")
        self.assertIn("venv", contenido, "venv/ debe estar en .gitignore.")

    def test_11_logo_sena_verde_oficial(self):
        """Verifica que el logo institucional sea el oficial en color verde (#39A900)."""
        ruta_logo = BASE_DIR / "web" / "sena_logo_verde.svg"
        self.assertTrue(ruta_logo.exists(), "El logo sena_logo_verde.svg debe existir.")
        contenido = ruta_logo.read_text(encoding="utf-8")
        self.assertIn("#39A900", contenido, "El logo debe tener el color verde institucional oficial #39A900.")


if __name__ == "__main__":
    unittest.main()

