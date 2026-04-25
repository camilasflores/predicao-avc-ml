import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # caminho local do dataset
    df = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

    # remoção de coluna sem valor preditivo
    df = df.drop(columns=["id"])

    # separação entre variáveis preditoras e alvo
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # identificação de colunas numéricas e categóricas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # pré-processamento
    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ),
    ])

    # split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # pipeline com SMOTE + regressão logística
    model = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # treino
    model.fit(X_train, y_train)

    # predição padrão
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== THRESHOLD PADRÃO (0.5) ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("=== Matriz de confusão ===")
    print(confusion_matrix(y_test, y_pred))

    # ajuste de threshold com recall mínimo desejado
    desired_recall = 0.7

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    valid_idxs = np.where(recall[:-1] >= desired_recall)[0]

    if len(valid_idxs) > 0:
        best_idx = valid_idxs[np.argmax(precision[valid_idxs])]
        best_threshold = thresholds[best_idx]

        print("\n=== THRESHOLD AJUSTADO ===")
        print("Threshold escolhido:", best_threshold)
        print("Recall:", recall[best_idx])
        print("Precision:", precision[best_idx])

        y_pred_adj = (y_proba >= best_threshold).astype(int)

        print("\n=== RESULTADO COM THRESHOLD AJUSTADO ===")
        print(classification_report(y_test, y_pred_adj))
        print("=== Matriz de confusão ajustada ===")
        print(confusion_matrix(y_test, y_pred_adj))
    else:
        print("\nNenhum threshold atingiu o recall desejado.")


if __name__ == "__main__":
    main()