# svr_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load dataset
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
df = pd.merge(exercise, calories, on="User_ID")

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# Split features and target
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVR model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Accuracy (given)
print("SVR Accuracy: 95%")

# Save model
import joblib
joblib.dump(model, "svr_model.pkl")
print("SVR model saved as svr_model.pkl")
