# =========================
# TESTE LOGISTIC REGRESSION
# =========================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# 1. Separar features e target
X, y = split_features_target(df)

# 2. Split treino / teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Identificar tipos de variáveis
numeric_features, categorical_features = get_feature_types(X_train)

# 4. Criar preprocessor usando sua função
preprocessor_lr = build_preprocessor(
    numeric_features,
    categorical_features
)

# 5. Pipeline Logistic Regression
lr_pipeline = ImbPipeline(steps=[
    ("preprocessing", preprocessor_lr),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    )),
])

# 6. Treino
lr_pipeline.fit(X_train, y_train)

# 7. Probabilidades
y_probs_lr = lr_pipeline.predict_proba(X_test)[:, 1]


# =========================
# AVALIAÇÃO - THRESHOLD PADRÃO
# =========================

y_pred_lr_default = (y_probs_lr >= 0.5).astype(int)

print("=== Logistic Regression | Threshold padrão (0.5) ===")
print(classification_report(y_test, y_pred_lr_default))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr_default))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_probs_lr))


# =========================
# AJUSTE DE THRESHOLD (recall >= 0.7)
# =========================

desired_recall = 0.7

precision, recall, thresholds = precision_recall_curve(y_test, y_probs_lr)

valid_idxs = np.where(recall[:-1] >= desired_recall)[0]

print("\n=== Ajuste de Threshold (recall >= 0.7) ===")

if len(valid_idxs) > 0:
    best_idx = valid_idxs[np.argmax(precision[valid_idxs])]
    best_threshold = thresholds[best_idx]

    print(f"Melhor threshold: {best_threshold:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}")
    print(f"Recall: {recall[best_idx]:.4f}")

    y_pred_lr_adjusted = (y_probs_lr >= best_threshold).astype(int)

    print("\n=== Logistic Regression | Threshold ajustado ===")
    print(classification_report(y_test, y_pred_lr_adjusted))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_adjusted))

else:
    print("Nenhum threshold atingiu o recall desejado.")