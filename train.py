"""
SkyFare Predictor - Training Pipeline (v2)
--------------------------------------------
Rebuilt from the original template with:
- Proper data cleaning (duplicates, nulls, inconsistent categories)
- Correct duration parsing (fixes the midnight-rollover bug in the old app.py)
- sklearn Pipeline (ColumnTransformer + Model) instead of manual if/elif encoding
- Model comparison: Linear Regression vs Random Forest vs Gradient Boosting
- 5-fold cross-validation on the best model
- Feature importance
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------
df = pd.read_excel("Flight Fare/Data_Train.xlsx")
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------
# Drop exact duplicate rows (220 found in EDA)
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# Drop the single row with missing Route/Total_Stops (not worth imputing for 1 row)
df = df.dropna(subset=["Total_Stops", "Route"])

# Fix inconsistent casing: 'No Info' vs 'No info'
df["Additional_Info"] = df["Additional_Info"].str.strip().replace({"No Info": "No info"})

# Drop the corrupted row where Duration is a nonsensical "5m" for a 2-stop flight
df = df[df["Duration"] != "5m"]

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

# --- Date of Journey ---
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
df["Journey_day"] = df["Date_of_Journey"].dt.day
df["Journey_month"] = df["Date_of_Journey"].dt.month
df["Journey_weekday"] = df["Date_of_Journey"].dt.weekday  # 0=Monday, useful signal, wasn't in original

# --- Departure time ---
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute

# --- Arrival time (raw, we only need hour/min for time-of-day feature) ---
# The Arrival_Time column sometimes has a trailing date e.g. "01:10 22 Mar" - take time part only
df["Arrival_hour"] = df["Arrival_Time"].apply(lambda x: int(x.split(" ")[0].split(":")[0]))
df["Arrival_min"] = df["Arrival_Time"].apply(lambda x: int(x.split(" ")[0].split(":")[1]))

# --- Duration: parsed correctly from the Duration STRING column ---
# This is the fix for the bug in the old app.py, which computed duration by
# subtracting Arrival_hour - Dep_hour directly, which breaks for overnight flights.
# Here we parse the duration the airline actually reported, so training and
# inference use the SAME correct source of truth.
def parse_duration(dur_str):
    hours = re.search(r"(\d+)h", dur_str)
    mins = re.search(r"(\d+)m", dur_str)
    h = int(hours.group(1)) if hours else 0
    m = int(mins.group(1)) if mins else 0
    return h, m

durations = df["Duration"].apply(parse_duration)
df["Duration_hours"] = durations.apply(lambda x: x[0])
df["Duration_mins"] = durations.apply(lambda x: x[1])
df["Duration_total_mins"] = df["Duration_hours"] * 60 + df["Duration_mins"]

# Sanity check: drop any flight with 0 total duration (impossible, would be more bad data)
before = len(df)
df = df[df["Duration_total_mins"] > 0]
print(f"Dropped {before - len(df)} rows with zero/invalid duration")

# --- Total_Stops: ordinal, not one-hot (non-stop < 1 stop < 2 stops... is a real order) ---
stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
df["Total_Stops"] = df["Total_Stops"].map(stops_map)

# --- Drop columns we don't use as model input ---
# Route is dropped because it's redundant with Source+Destination+Stops (high-cardinality, would overfit)
# Additional_Info dropped for the baseline model - see notes at bottom for why (90%+ is "No info")
df_model = df.drop(columns=[
    "Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration",
    "Route", "Additional_Info"
])

print(f"\nFinal cleaned shape: {df_model.shape}")
print(df_model.head())

# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X = df_model.drop(columns=["Price"])
y = df_model["Price"]

categorical_cols = ["Airline", "Source", "Destination"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------
# 5. PIPELINE (ColumnTransformer handles encoding - no manual if/elif!)
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"  # numeric_cols pass through untouched
)

# ---------------------------------------------------------
# 6. MODEL COMPARISON
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    # max_depth/min_samples_leaf capped on purpose: an uncapped forest scored
    # slightly worse (RMSE 1917 vs 1834) AND produced a 98MB pickle - capping
    # depth here reduces overfitting on noisy leaves AND shrinks the file
    # well under GitHub's 100MB push limit.
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    # NOTE: run compare_xgboost.py separately (needs `pip install xgboost`)
    # to add XGBoost to this table - not available in this sandboxed environment.
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)})
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------
# 7. CROSS-VALIDATION on best model
# ---------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
cv_rmse = -cv_scores

print("\n" + "=" * 60)
print(f"5-FOLD CROSS VALIDATION ({best_model_name})")
print("=" * 60)
print(f"Fold RMSEs: {np.round(cv_rmse, 2)}")
print(f"Mean RMSE: {cv_rmse.mean():.2f}  |  Std: {cv_rmse.std():.2f}")

# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE (best tree-based model)
# ---------------------------------------------------------
if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(cat_feature_names) + numeric_cols

    importances = best_pipe.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    print(fi_df.head(10).to_string(index=False))
    fi_df.to_csv("results/feature_importance.csv", index=False)

# ---------------------------------------------------------
# 9. SAVE ARTIFACTS
# ---------------------------------------------------------
results_df.to_csv("results/model_comparison.csv", index=False)

with open("flight_price_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

print("\nSaved: model_comparison.csv, feature_importance.csv, flight_price_pipeline.pkl")
