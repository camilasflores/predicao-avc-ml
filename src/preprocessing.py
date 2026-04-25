from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    """
    return pd.read_csv(file_path)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas que não agregam valor preditivo ao modelo.
    Exemplo: 'id'.
    """
    df = df.copy()

    columns_to_drop = [col for col in ["id"] if col in df.columns]
    return df.drop(columns=columns_to_drop)


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "stroke",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa variáveis preditoras (X) e variável alvo (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"A coluna alvo '{target_column}' não existe no DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifica colunas numéricas e categóricas.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Cria o pipeline de pré-processamento:
    - Numéricas: imputação pela mediana + padronização
    - Categóricas: imputação pela moda + one-hot encoding
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor