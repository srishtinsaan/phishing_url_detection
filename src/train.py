import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load dataset
df = pd.read_csv("../data/phishing_dataset.csv")

# convert labels to numbers
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# drop URL column (not needed)
df = df.drop(columns=['url'])

# split features & target
X = df.drop(columns=['status'])
y = df['status']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# save model
joblib.dump(model, "../models/model.pkl")

print("\nModel trained & saved!")