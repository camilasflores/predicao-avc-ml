# =========================
# TESTE GRADIENT BOOSTING
# =========================

# 1. Split treino / teste
from sklearn.model_selection import train_test_split

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2. Tipos de variáveis
numeric_features, categorical_features = get_feature_types(X_train)

# 3. Preprocessor
preprocessor_gb = build_preprocessor(
    numeric_features,
    categorical_features
)

# 4. Pipeline Gradient Boosting
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

gb_pipeline = ImbPipeline(steps=[
    ("preprocessing", preprocessor_gb),
    ("smote", SMOTE(random_state=42)),
    ("model", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )),
])

# 5. Treino
gb_pipeline.fit(X_train, y_train)

# 6. Probabilidades
y_probs_gb = gb_pipeline.predict_proba(X_test)[:, 1]

# =========================
# AVALIAÇÃO - THRESHOLD PADRÃO
# =========================
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred_gb_default = (y_probs_gb >= 0.5).astype(int)

print("=== Gradient Boosting | Threshold padrão (0.5) ===")
print(classification_report(y_test, y_pred_gb_default))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb_default))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_probs_gb))


# =========================
# AJUSTE DE THRESHOLD (recall >= 0.7)
# =========================
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_probs_gb)

results = []

for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    if r >= 0.7:
        results.append((t, p, r))

print("\n=== Ajuste de Threshold (recall >= 0.7) ===")

if results:
    best_threshold, best_precision, best_recall = max(results, key=lambda x: x[1])

    print(f"Melhor threshold: {best_threshold:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")

    # Aplicar threshold ajustado
    y_pred_gb_adjusted = (y_probs_gb >= best_threshold).astype(int)

    print("\n=== Gradient Boosting | Threshold ajustado ===")
    print(classification_report(y_test, y_pred_gb_adjusted))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_gb_adjusted))

else:
    print("Nenhum threshold atingiu recall >= 0.7")