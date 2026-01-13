import pandas as pd

df = pd.read_csv("results/phase4_channel_datasets/C4-P4.csv")

# List all object (string) columns
object_cols = df.select_dtypes(include="object").columns.tolist()
print("Object columns:", object_cols)

# Inspect unique values for each
for col in object_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(dropna=False))
