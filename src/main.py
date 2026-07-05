"""Ejecución completa del proyecto LOOCV Diabetes.

Uso desde la raíz del proyecto:
    python src/main.py

El flujo replica la arquitectura de la plantilla base:
datos → validación → transformación → modelado LOOCV → evaluación →
visualización → reporte → persistencia de resultados.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dataset import configure_logging, load_dataset, split_features_target, summarize_dataset
from modeling import build_models, evaluate_predictions, run_loocv_predictions
from reporting import build_feature_importance_table, print_metrics_report, save_table
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix_from_predictions,
    plot_metrics_comparison,
    plot_random_forest_feature_importance,
)


LOGGER = logging.getLogger(__name__)


def get_project_paths() -> dict[str, Path]:
    """Define rutas principales del proyecto.

    Returns:
        Diccionario con rutas de datos y salidas.
    """
    project_root = Path(__file__).resolve().parents[1]
    return {
        "root": project_root,
        "data": project_root / "data" / "diabetes.csv",
        "tables": project_root / "outputs" / "tablas",
        "figures": project_root / "outputs" / "graficas",
    }


def run_pipeline() -> None:
    """Ejecuta el pipeline completo de análisis supervisado con LOOCV.

    Returns:
        None.
    """
    configure_logging()
    paths = get_project_paths()

    LOGGER.info("Iniciando pipeline LOOCV Diabetes.")
    data = load_dataset(paths["data"])
    X, y = split_features_target(data)

    summary = summarize_dataset(data)
    save_table(summary.reset_index().rename(columns={"index": "Variable"}), paths["tables"] / "resumen_dataset.csv")
    plot_class_distribution(data, paths["figures"] / "distribucion_clases_dataset.png")

    models = build_models(random_state=42)
    predictions = run_loocv_predictions(models=models, X=X, y=y, n_jobs=1)
    metrics_df = evaluate_predictions(predictions)

    save_table(predictions, paths["tables"] / "predicciones_loocv_cache.csv")
    save_table(metrics_df, paths["tables"] / "metricas_loocv_modelos.csv")
    print_metrics_report(metrics_df)

    for model_name in models:
        file_name = model_name.lower().replace(" ", "_").replace("í", "i")
        plot_confusion_matrix_from_predictions(
            predictions=predictions,
            model_name=model_name,
            output_path=paths["figures"] / f"matriz_confusion_{file_name}.png",
        )

    plot_metrics_comparison(metrics_df, paths["figures"] / "comparacion_metricas_loocv.png")

    random_forest_model = models["Random Forest"]
    random_forest_model.fit(X, y)
    importance_df = build_feature_importance_table(random_forest_model, list(X.columns))
    save_table(importance_df, paths["tables"] / "importancia_variables_random_forest.csv")
    plot_random_forest_feature_importance(
        importance_df,
        paths["figures"] / "importancia_variables_random_forest.png",
    )

    LOGGER.info("Pipeline LOOCV Diabetes finalizado correctamente.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        LOGGER.exception("El pipeline falló: %s", error)
        raise
