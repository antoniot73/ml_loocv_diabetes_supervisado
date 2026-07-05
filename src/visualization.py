"""Visualización de resultados del pipeline LOOCV."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


LOGGER = logging.getLogger(__name__)


def ensure_directory(directory: Path) -> None:
    """Crea un directorio si no existe.

    Args:
        directory: Ruta del directorio.
    """
    directory.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Grafica la distribución de clases de la variable Outcome.

    Args:
        data: Dataset de diabetes.
        output_path: Ruta de salida PNG.
    """
    ensure_directory(output_path.parent)
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=data, x="Outcome")
    ax.set_title("Distribución de clases del dataset de diabetes")
    ax.set_xlabel("Clase Outcome")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Gráfica guardada: %s.", output_path)


def plot_confusion_matrix_from_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    """Grafica matriz de confusión para un modelo.

    Args:
        predictions: DataFrame con y_true y columnas de predicción.
        model_name: Nombre de la columna del modelo.
        output_path: Ruta de salida PNG.
    """
    ensure_directory(output_path.parent)

    if model_name not in predictions.columns:
        raise ValueError(f"No existe la columna del modelo: {model_name}")

    cm = confusion_matrix(predictions["y_true"], predictions[model_name], labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No diabetes", "Diabetes"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusión LOOCV - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Matriz de confusión guardada: %s.", output_path)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Grafica comparación de métricas principales por modelo.

    Args:
        metrics_df: Tabla comparativa de métricas.
        output_path: Ruta de salida PNG.
    """
    ensure_directory(output_path.parent)
    metric_columns = [
        "Precisión Global",
        "Error Global",
        "Precisión Positiva (PP)",
        "Precisión Negativa (PN)",
        "Asertividad Positiva (AP)",
        "Asertividad Negativa (NP)",
    ]

    long_df = metrics_df.melt(
        id_vars="Modelo",
        value_vars=metric_columns,
        var_name="Métrica",
        value_name="Valor",
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=long_df, x="Métrica", y="Valor", hue="Modelo")
    ax.set_title("Comparación de métricas LOOCV por modelo")
    ax.set_xlabel("Métrica")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=35)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Comparación de métricas guardada: %s.", output_path)


def plot_random_forest_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grafica importancia de variables para Random Forest.

    Args:
        feature_importance: DataFrame con columnas variable e importancia.
        output_path: Ruta de salida PNG.
    """
    ensure_directory(output_path.parent)

    required_columns = {"Variable", "Importancia"}
    if not required_columns.issubset(feature_importance.columns):
        raise ValueError("feature_importance debe incluir Variable e Importancia.")

    ordered = feature_importance.sort_values("Importancia", ascending=False)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=ordered, y="Variable", x="Importancia")
    ax.set_title("Importancia de variables - Random Forest")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Importancia de variables guardada: %s.", output_path)
