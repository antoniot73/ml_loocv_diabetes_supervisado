"""Modelado supervisado con LOOCV.

Este módulo implementa Random Forest, SVM lineal y Red Neuronal MLP con
validación cruzada Leave-One-Out. Se calcula el conjunto de métricas pedido:
Precisión Global, Error Global, PP, PN, FP, FN, AP y NP.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LOGGER = logging.getLogger(__name__)


def build_models(random_state: int = 42) -> "OrderedDict[str, BaseEstimator]":
    """Construye los modelos predictivos supervisados.

    Args:
        random_state: Semilla aleatoria para reproducibilidad.

    Returns:
        Diccionario ordenado con los modelos configurados.
    """
    if not isinstance(random_state, int):
        raise TypeError("random_state debe ser entero.")

    models: "OrderedDict[str, BaseEstimator]" = OrderedDict()

    models["Random Forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["SVM Lineal"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=5000,
                    dual=False,
                ),
            ),
        ]
    )

    models["Red Neuronal MLP"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(12,),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=400,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    LOGGER.info("Modelos construidos: %s.", list(models.keys()))
    return models


def validate_features_target(X: pd.DataFrame, y: pd.Series) -> None:
    """Valida entradas para entrenamiento y evaluación.

    Args:
        X: Variables predictoras.
        y: Variable objetivo.

    Raises:
        ValueError: Si las dimensiones o clases no son válidas.
    """
    if X.empty:
        raise ValueError("X no puede estar vacío.")

    if y.empty:
        raise ValueError("y no puede estar vacío.")

    if len(X) != len(y):
        raise ValueError("X e y deben tener el mismo número de filas.")

    if y.nunique() != 2:
        raise ValueError("LOOCV configurado para clasificación binaria.")


def run_loocv_predictions(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Ejecuta LOOCV y devuelve predicciones por modelo.

    Args:
        models: Diccionario de modelos.
        X: Variables predictoras.
        y: Variable objetivo.
        n_jobs: Número de trabajos paralelos usado por cross_val_predict.

    Returns:
        DataFrame con y real y predicciones LOOCV.

    Raises:
        RuntimeError: Si un modelo falla durante LOOCV.
    """
    validate_features_target(X, y)

    if not models:
        raise ValueError("Debe proporcionarse al menos un modelo.")

    loo = LeaveOneOut()
    predictions = pd.DataFrame({"y_true": y.to_numpy(dtype=int)})

    for model_name, model in models.items():
        LOGGER.info("Ejecutando LOOCV para: %s.", model_name)
        try:
            predictions[model_name] = cross_val_predict(
                estimator=model,
                X=X,
                y=y,
                cv=loo,
                n_jobs=n_jobs,
                method="predict",
            ).astype(int)
        except Exception as exc:
            raise RuntimeError(f"Falló LOOCV para el modelo {model_name}.") from exc

    LOGGER.info("Predicciones LOOCV generadas correctamente.")
    return predictions


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de clasificación binaria solicitadas.

    Convención de matriz de confusión:
        TN, FP, FN, TP.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Diccionario con Precisión Global, Error Global, PP, PN, FP, FN, AP y NP.
    """
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    total = tn + fp + fn + tp

    if total == 0:
        raise ValueError("No hay observaciones para calcular métricas.")

    precision_global = (tp + tn) / total
    error_global = (fp + fn) / total
    precision_positiva = tp / (tp + fn) if (tp + fn) else 0.0
    precision_negativa = tn / (tn + fp) if (tn + fp) else 0.0
    asertividad_positiva = tp / (tp + fp) if (tp + fp) else 0.0
    asertividad_negativa = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "Precisión Global": precision_global,
        "Error Global": error_global,
        "Precisión Positiva (PP)": precision_positiva,
        "Precisión Negativa (PN)": precision_negativa,
        "Falsos Positivos (FP)": int(fp),
        "Falsos Negativos (FN)": int(fn),
        "Asertividad Positiva (AP)": asertividad_positiva,
        "Asertividad Negativa (NP)": asertividad_negativa,
        "TN": int(tn),
        "TP": int(tp),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evalúa las predicciones LOOCV por modelo.

    Args:
        predictions: DataFrame con columna y_true y columnas de modelos.

    Returns:
        DataFrame comparativo de métricas.
    """
    if "y_true" not in predictions.columns:
        raise ValueError("predictions debe contener la columna y_true.")

    rows = []
    y_true = predictions["y_true"].to_numpy(dtype=int)

    for model_name in predictions.columns.drop("y_true"):
        y_pred = predictions[model_name].to_numpy(dtype=int)
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics["Modelo"] = model_name
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    ordered_columns = [
        "Modelo",
        "Precisión Global",
        "Error Global",
        "Precisión Positiva (PP)",
        "Precisión Negativa (PN)",
        "Falsos Positivos (FP)",
        "Falsos Negativos (FN)",
        "Asertividad Positiva (AP)",
        "Asertividad Negativa (NP)",
        "TN",
        "TP",
    ]
    return metrics_df[ordered_columns]
