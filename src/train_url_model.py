import pandas as pd
import joblib
from features import extract_features

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

from catboost import CatBoostClassifier

# 🔹 Load dataset
df = pd.read_csv("../data/phishing_dataset.csv")

# 🔹 Clean URLs
df['url'] = df['url'].str.lower()

# 🔹 Convert labels
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# 🔹 Extract features
df['features'] = df['url'].apply(extract_features)

# 🔹 Convert to DataFrame
X = pd.DataFrame(df['features'].tolist())
y = df['status']

print("Number of features:", X.shape[1])

# 🔹 Train-test split (STRATIFIED 🔥)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Base model
base_model = CatBoostClassifier(verbose=0)

# 🔹 Hyperparameter space
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'iterations': [300, 500, 800, 1000],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# 🔹 Randomized Search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',   # you can change to 'roc_auc'
    cv=3,
    verbose=1,
    n_jobs=-1
)

# 🔹 Train
random_search.fit(X_train, y_train)

# 🔹 Best model
best_model = random_search.best_estimator_

print("\nBest Parameters:", random_search.best_params_)

# 🔹 Evaluate
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# 🔹 Save model
joblib.dump(best_model, "../models/url_model.pkl")

# 🔹 Save feature count
joblib.dump(len(X.columns), "../models/feature_count.pkl")

print("\nTuned model trained & saved successfully!")