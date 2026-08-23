import os
import pandas as pd

DATA_PATH = os.path.join('tourism_project', 'data', 'tourism.csv')

EXPECTED_COLUMNS = [
    "CustomerID","ProdTaken","Age","TypeofContact","CityTier",
    "DurationOfPitch","Occupation","Gender","NumberOfPersonVisiting",
    "NumberOfFollowups","ProductPitched","PreferredPropertyStar","MaritalStatus",
    "NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar",
    "NumberOfChildrenVisiting","Designation","MonthlyIncome"
]

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"ERROR reading CSV: {e}")
    exit(1)

cols = list(df.columns)
missing = [c for c in EXPECTED_COLUMNS if c not in cols]
extra = [c for c in cols if c not in EXPECTED_COLUMNS]

print(f"Loaded {DATA_PATH} — rows: {len(df)}, columns: {len(cols)}")
if missing:
    print("Missing expected columns:")
    for c in missing:
        print(f" - {c}")
    exit(1)
else:
    print("All expected columns present.")

if extra:
    print("Extra columns found (unexpected):")
    for c in extra:
        print(f" - {c}")

print("\nColumn types and missing values:")
print(df.dtypes)
print(df.isnull().sum())

if 'ProdTaken' in df.columns:
    if not set(df['ProdTaken'].dropna().unique()).issubset({0,1}):
        print("WARNING: 'ProdTaken' contains values other than 0 and 1")

print("Data registration successful.")
