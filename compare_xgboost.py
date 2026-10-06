"""
Run this LOCALLY in VS Code (needs internet to install xgboost):
    pip install xgboost
    python compare_xgboost.py

Adds XGBoost to the model comparison table alongside the models already
tested in train.py (Linear Regression, Random Forest, Gradient Boosting).
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --- same cleaning/feature engineering as train.py ---
df = pd.read_excel("Data_Train.xlsx")  # adjust path to your local file
df = df.drop_duplicates()
df = df.dropna(subset=["Total_Stops", "Route"])
df["Additional_Info"] = df["Additional_Info"].str.strip().replace({"No Info": "No info"})
df = df[df["Duration"] != "5m"]

df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
df["Journey_day"] = df["Date_of_Journey"].dt.day
df["Journey_month"] = df["Date_of_Journey"].dt.month
df["Journey_weekday"] = df["Date_of_Journey"].dt.weekday

df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df["Arrival_hour"] = df["Arrival_Time"].apply(lambda x: int(x.split(" ")[0].split(":")[0]))
df["Arrival_min"] = df["Arrival_Time"].apply(lambda x: int(x.split(" ")[0].split(":")[1]))

def parse_duration(dur_str):
    hours = re.search(r"(\d+)h", dur_str)
    mins = re.search(r"(\d+)m", dur_str)
    return (int(hours.group(1)) if hours else 0, int(mins.group(1)) if mins else 0)

durations = df["Duration"].apply(parse_duration)
df["Duration_hours"] = durations.apply(lambda x: x[0])
df["Duration_mins"] = durations.apply(lambda x: x[1])
df["Duration_total_mins"] = df["Duration_hours"] * 60 + df["Duration_mins"]
df = df[df["Duration_total_mins"] > 0]

stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
df["Total_Stops"] = df["Total_Stops"].map(stops_map)

df_model = df.drop(columns=["Date_of_Journey", "Dep_Time", "Arrival_Time",
                             "Duration", "Route", "Additional_Info"])

X = df_model.drop(columns=["Price"])
y = df_model["Price"]
categorical_cols = ["Airline", "Source", "Destination"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

xgb_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
])

xgb_pipe.fit(X_train, y_train)
preds = xgb_pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"XGBoost -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
print("\nCompare this row against model_comparison.csv (from train.py) to complete your table.")
