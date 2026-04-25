from __future__ import annotations

import pandas as pd


def create_risk_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a feature 'risk_count' com base na soma de fatores de risco:
    - hipertensão
    - doença cardíaca
    - idade acima de 50
    - glicose acima da mediana
    - fumante atual ou ex-fumante
    """
    df = df.copy()

    df["is_smoker"] = (
        df["smoking_status"].isin(["smokes", "formerly smoked"]).astype(int)
    )
    df["high_glucose"] = (df["avg_glucose_level"] > 100).astype(int)
    df["age_risk"] = (df["age"] > 50).astype(int)

    df["risk_count"] = (
        df["hypertension"].astype(int)
        + df["heart_disease"].astype(int)
        + df["is_smoker"]
        + df["high_glucose"]
        + df["age_risk"]
    )

    return df
