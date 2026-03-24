import pandas as pd
import joblib
from features import extract_features

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

from catboost import CatBoostClassifier

df = pd.read_csv("../data/phishing_dataset.csv")

df['url'] = df['url'].str.lower()

df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

df['features'] = df['url'].apply(extract_features)

X = pd.DataFrame(df['features'].tolist())
y = df['status']

print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_model = CatBoostClassifier(verbose=0)

param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'iterations': [300, 500, 800, 1000],
    'l2_leaf_reg': [1, 3, 5, 7]
}

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',   
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\nBest Parameters:", random_search.best_params_)

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

joblib.dump(best_model, "../models/url_model.pkl")

joblib.dump(len(X.columns), "../models/feature_count.pkl")

print("\nTuned model trained & saved successfully!")