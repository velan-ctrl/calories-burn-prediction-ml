# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------
# Step 1: Load dataset
# ------------------------
exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

# Merge
df = pd.merge(exercise, calories, on="User_ID")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Features & Target
X = df[['age','gender','height','weight','duration','heart_rate','body_temp']]
y = df['calories']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 2: Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train Models
ada = AdaBoostRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
svr = SVR(kernel='rbf').fit(X_train, y_train)
dt  = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# Predictions
y_pred_ada = ada.predict(X_test)
y_pred_svr = svr.predict(X_test)
y_pred_dt  = dt.predict(X_test)

# Step 4: Regression Metrics
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'R2_percent': r2*100}

print("AdaBoost:", regression_metrics(y_test, y_pred_ada))
print("SVR:", regression_metrics(y_test, y_pred_svr))
print("DecisionTree:", regression_metrics(y_test, y_pred_dt))

# Step 5: Save Best Model + Scaler
results = {
    "AdaBoost": mean_absolute_error(y_test, y_pred_ada),
    "SVR": mean_absolute_error(y_test, y_pred_svr),
    "DecisionTree": mean_absolute_error(y_test, y_pred_dt)
}
best = min(results, key=results.get)
best_model = {"AdaBoost": ada, "SVR": svr, "DecisionTree": dt}[best]

joblib.dump(best_model, f"best_model_{best}.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"\n✅ Best model: {best} saved as best_model_{best}.pkl")
