"""Reporte y persistencia de resultados del pipeline LOOCV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def ensure_directory(directory: Path) -> None:
    """Crea un directorio de salida si no existe.

    Args:
        directory: Ruta del directorio.
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_table(data: pd.DataFrame, output_path: Path) -> None:
    """Guarda una tabla en CSV con validaciones básicas.

    Args:
        data: DataFrame a guardar.
        output_path: Ruta de salida.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data debe ser un pandas.DataFrame.")

    ensure_directory(output_path.parent)
    data.to_csv(output_path, index=False, encoding="utf-8")
    LOGGER.info("Tabla guardada: %s.", output_path)


def print_metrics_report(metrics_df: pd.DataFrame) -> None:
    """Imprime reporte comparativo en consola.

    Args:
        metrics_df: Tabla de métricas por modelo.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df no puede estar vacío.")

    print("\n================ REPORTE COMPARATIVO LOOCV ================\n")
    print(metrics_df.to_string(index=False))
    print("\n============================================================\n")


def build_feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Construye tabla de importancia de variables para Random Forest.

    Args:
        model: Pipeline ajustado con un estimador RandomForestClassifier.
        feature_names: Nombres de variables predictoras.

    Returns:
        DataFrame ordenado por importancia.
    """
    if not feature_names:
        raise ValueError("feature_names no puede estar vacío.")

    estimator = model.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        raise ValueError("El modelo no contiene feature_importances_.")

    importance_df = pd.DataFrame(
        {
            "Variable": feature_names,
            "Importancia": estimator.feature_importances_,
        }
    ).sort_values("Importancia", ascending=False)

    return importance_df
