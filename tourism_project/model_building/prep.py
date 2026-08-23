import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_CSV = os.path.join('tourism_project','data','tourism.csv')

if not os.path.exists(DATA_CSV):
    print(f"Error: {DATA_CSV} not found. Please ensure the data registration step runs correctly.")
    exit(1)

df = pd.read_csv(DATA_CSV)
# normalize column names
df.columns = [c.strip() for c in df.columns]

# quick cleaning fixes (common typos in dataset)
if 'Fe Male' in df.columns or any(df[col].astype(str).str.contains('Fe Male').any() for col in df.columns if df[col].dtype==object):
  df = df.replace({'Fe Male':'Female'})

print("Dataset shape before cleaning:", df.shape)

# Feature selection / cleanup
drop_cols = ['CustomerID'] if 'CustomerID' in df.columns else []
df = df.drop(columns=drop_cols)

# Ensure target is integer
df['ProdTaken'] = df['ProdTaken'].astype(int)

# Simple missing value strategy: drop rows with missing target; fill numeric NAs with median
df = df[~df['ProdTaken'].isna()].copy()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'ProdTaken']
for c in num_cols:
  if df[c].isnull().any():
    df[c] = df[c].fillna(df[c].median())

# Fill categorical NAs with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
  df[c] = df[c].fillna('Unknown')

print("Dataset shape after cleaning:", df.shape)

# Train/test split
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits for reproducibility / later cells
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

print('Saved splits: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv')
print('Data preparation complete.')
