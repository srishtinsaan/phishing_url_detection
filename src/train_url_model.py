import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from features import extract_features
from catboost import CatBoostClassifier

# 🔹 Load dataset
df = pd.read_csv("../data/phishing_dataset.csv")

# 🔹 Clean URLs
df['url'] = df['url'].str.lower()

# 🔹 Convert labels
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# 🔹 Extract features from URL
df['features'] = df['url'].apply(extract_features)

# 🔹 Convert list → dataframe
X = pd.DataFrame(df['features'].tolist())
y = df['status']

# 🔹 Debug: print feature count
print("Number of features:", X.shape[1])

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Train model (optimized CatBoost)
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0
)

model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# 🔹 Save model
joblib.dump(model, "../models/url_model.pkl")

# 🔹 Save feature count (for UI safety)
joblib.dump(len(X.columns), "../models/feature_count.pkl")

print("URL-based model trained & saved successfully!")