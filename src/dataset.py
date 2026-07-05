"""Carga y validación del dataset de diabetes.

Este módulo separa las responsabilidades de entrada y validación de datos.
La estructura replica el pipeline base de la práctica KNN/CART, ajustada al
dataset de diabetes y a la técnica LOOCV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


EXPECTED_COLUMNS = {
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
}


def configure_logging(level: int = logging.INFO) -> None:
    """Configura la bitácora del pipeline.

    Args:
        level: Nivel mínimo de logging.

    Returns:
        None.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_file_path(file_path: Path) -> None:
    """Valida que la ruta del dataset exista y sea un archivo CSV.

    Args:
        file_path: Ruta del archivo CSV.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la extensión no es CSV.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path debe ser un objeto pathlib.Path.")

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("El archivo de entrada debe tener extensión .csv.")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Carga el dataset de diabetes desde CSV.

    Args:
        file_path: Ruta del archivo diabetes.csv.

    Returns:
        DataFrame con los registros cargados.

    Raises:
        RuntimeError: Si ocurre un error durante la carga.
    """
    validate_file_path(file_path)

    try:
        data = pd.read_csv(file_path)
        LOGGER.info("Dataset cargado correctamente con forma %s.", data.shape)
        return data
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError("El archivo CSV está vacío.") from exc
    except pd.errors.ParserError as exc:
        raise RuntimeError("El archivo CSV no pudo analizarse correctamente.") from exc


def validate_dataset(data: pd.DataFrame, target_column: str = "Outcome") -> None:
    """Valida estructura mínima, columnas, nulos y variable objetivo.

    Args:
        data: Dataset cargado.
        target_column: Nombre de la variable objetivo.

    Raises:
        TypeError: Si data no es DataFrame.
        ValueError: Si faltan columnas, datos o clases.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data debe ser un pandas.DataFrame.")

    if data.empty:
        raise ValueError("El dataset está vacío.")

    missing_columns = EXPECTED_COLUMNS.difference(set(data.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas esperadas: {sorted(missing_columns)}")

    if target_column not in data.columns:
        raise ValueError(f"No existe la variable objetivo: {target_column}")

    if data[target_column].nunique() != 2:
        raise ValueError("La variable objetivo debe tener exactamente dos clases.")

    numeric_columns = data.columns.drop(target_column)
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"La columna {column} debe ser numérica.")

    LOGGER.info("Validación del dataset completada correctamente.")


def split_features_target(
    data: pd.DataFrame,
    target_column: str = "Outcome",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y variable objetivo.

    Args:
        data: Dataset validado.
        target_column: Nombre de la variable objetivo.

    Returns:
        Tupla con matriz X y vector y.
    """
    validate_dataset(data, target_column=target_column)
    features = data.drop(columns=[target_column])
    target = data[target_column].astype(int)
    LOGGER.info("Separación X/y completada: X=%s, y=%s.", features.shape, target.shape)
    return features, target


def summarize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen estadístico exportable del dataset.

    Args:
        data: Dataset validado.

    Returns:
        DataFrame con resumen estadístico transpuesto.
    """
    validate_dataset(data)
    summary = data.describe(include="all").T
    summary["missing_values"] = data.isna().sum()
    return summary
