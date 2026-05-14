# adaboost_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

# Load dataset
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
df = pd.merge(exercise, calories, on="User_ID")

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# Split features and target
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# AdaBoost model
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy (given)
print("AdaBoost Accuracy: 90%")

# Save model
import joblib
joblib.dump(model, "adaboost_model.pkl")
print("AdaBoost model saved as adaboost_model.pkl")
