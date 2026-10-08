"""
Script Orquestador Principal: main.py
Ejecuta el pipeline completo de analítica de datos e integración de IA generativa.
Evidencia: AA2-EV03 - Video Configuración de Analítica de Datos
Programa: Aplicación de la Inteligencia Artificial en la Integración de Datos
SENA - 2026
"""

import sys
import time
import importlib
from pathlib import Path

# Configurar directorio base
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Carga dinámica de módulos (permite nombres con prefijo numérico como 01_ingesta)
mod_ingesta = importlib.import_module("src.01_ingesta")
mod_limpieza = importlib.import_module("src.02_limpieza")
mod_eda = importlib.import_module("src.03_eda")
mod_modelado = importlib.import_module("src.04_modelado")
mod_ia_gen = importlib.import_module("src.05_ia_generativa")
mod_visualizacion = importlib.import_module("src.06_visualizacion")


def imprimir_encabezado():
    print("=" * 75)
    print("🚀 PIPELINE DE ANALÍTICA DE DATOS CON IA GENERATIVA (ANTIGRAVITY)")
    print("   Evidencia SENA: AA2-EV03 | Resultado de Aprendizaje: 220501115-02")
    print("   Dataset: Titanic - Machine Learning from Disaster")
    print("=" * 75)


def imprimir_separador(fase: str, numero: int):
    print("\n" + "#" * 75)
    print(f"▶ ETAPA {numero}: {fase.upper()}")
    print("#" * 75)


def main():
    inicio_total = time.time()
    imprimir_encabezado()

    try:
        # FASE 1: Ingesta de datos
        imprimir_separador("Ingesta de Datos Crudos y Auditoría Inicial", 1)
        mod_ingesta.ejecutar_ingesta()

        # FASE 2: Limpieza e Imputación
        imprimir_separador("Limpieza Estadística, Imputación y Feature Engineering", 2)
        mod_limpieza.ejecutar_limpieza()

        # FASE 3: Análisis Exploratorio y Segmentaciones
        imprimir_separador("Análisis Exploratorio de Datos (EDA) y Segmentaciones", 3)
        mod_eda.ejecutar_eda()

        # FASE 4: Modelado de Machine Learning (K-Means)
        imprimir_separador("Modelado de Machine Learning no Supervisado (K-Means)", 4)
        mod_modelado.ejecutar_modelado()

        # FASE 5: Interpretación con IA Generativa
        imprimir_separador("Interpretación Analítica de Negocio con IA Generativa", 5)
        mod_ia_gen.ejecutar_ia_generativa()

        # FASE 6: Visualizaciones Consolidadas
        imprimir_separador("Generación de Dashboard Visual Consolidado", 6)
        mod_visualizacion.ejecutar_visualizaciones()

        tiempo_total = time.time() - inicio_total
        print("\n" + "=" * 75)
        print(f"🎉 PIPELINE COMPLETADO EXITOSAMENTE en {tiempo_total:.2f} segundos.")
        print("📁 Artefactos generados:")
        print("   - Datos limpios:   data/processed/titanic_clean.csv")
        print("   - Figuras y plots: outputs/figuras/ (5 gráficos + 1 dashboard)")
        print("   - Reporte CSV:     outputs/reportes/resumen_clusters.csv")
        print("   - Informe IA:      outputs/reportes/interpretacion_ia.txt")
        print("=" * 75)

    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución del pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

