import pandas as pd

df = pd.read_csv("../data/phishing_dataset.csv")

print(df.head())
print("\nColumns:\n", df.columns)