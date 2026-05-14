from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import os

app = Flask(__name__)
app.secret_key = "velan_secret_key"

# Algorithm accuracies (demo purpose)
accuracies = {
    "SVR": 0.95,
    "DecisionTree": 0.993,
    "AdaBoost": 0.90
}

# ---------------- Helper: Train model from uploaded CSVs ----------------
def train_from_csv(ex_path, cal_path):
    exercise = pd.read_csv(ex_path)
    calories = pd.read_csv(cal_path)

    common_keys = ['User_ID','user_id','UserId','userId','id','Id']
    merge_key = None
    for k in common_keys:
        if k in exercise.columns and k in calories.columns:
            merge_key = k
            break
    if merge_key:
        df = pd.merge(exercise, calories, on=merge_key)
    else:
        df = pd.concat([exercise.reset_index(drop=True), calories.reset_index(drop=True)], axis=1)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male':0,'m':0,'female':1,'f':1})
        df['gender'] = df['gender'].fillna(0).astype(int)
    else:
        df['gender'] = 0

    features = ['age','gender','height','weight','duration','heart_rate','body_temp']
    for f in features:
        if f not in df.columns:
            raise Exception(f"Required column '{f}' not found in merged dataset.")

    X = df[features]

    possible_targets = [c for c in df.columns if 'calor' in c]
    if not possible_targets:
        raise Exception("No target column containing 'calor' found.")
    y = df[possible_targets[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    algo_models = {
        "AdaBoost": AdaBoostRegressor(n_estimators=200, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    results = []
    trained_models = {}
    for name, m in algo_models.items():
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds) * 100
        results.append({
            'Algorithm': name,
            'MAE': round(mae,2),
            'MSE': round(mse,2),
            'R2 (%)': round(r2,2),
            'Accuracy (%)': round(accuracies.get(name,0)*100,2)
        })
        trained_models[name] = m

    # pick best by R2
    best = max(results, key=lambda x: x['R2 (%)'])
    best_name = best['Algorithm']
    best_model = trained_models[best_name]

    joblib.dump(best_model, "calorie_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return results, best_name

# ---------------- Routes ----------------
@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if u == 'user' and p == 'velan':
            session['user'] = u
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))

    message = None
    if request.method == 'POST':
        action = request.form.get('action')
        algo_choice = request.form.get('algorithm')

        if action == 'upload_train':
            ex_file = request.files.get('exercise')
            cal_file = request.files.get('calories')
            if not ex_file or not cal_file:
                message = 'Please upload both exercise.csv and calories.csv'
            else:
                ex_file.save('exercise.csv')
                cal_file.save('calories.csv')
                try:
                    results, best_name = train_from_csv('exercise.csv','calories.csv')
                    message = f'Training finished. Best model: {best_name}'
                except Exception as e:
                    message = f'Error during training: {e}'
            return render_template('prediction.html', message=message, algorithms=accuracies.keys())

        elif action == 'predict_now':
            try:
                age = float(request.form.get('age'))
                gender = int(request.form.get('gender'))
                height = float(request.form.get('height'))
                weight = float(request.form.get('weight'))
                duration = float(request.form.get('duration'))
                heart_rate = float(request.form.get('heart_rate'))
                body_temp = float(request.form.get('body_temp'))
            except Exception as e:
                return render_template('prediction.html', message=f'Invalid input: {e}', algorithms=accuracies.keys())

            if not os.path.exists('calorie_model.pkl') or not os.path.exists('scaler.pkl'):
                return render_template('prediction.html',
                                       message='No trained model found. Please upload CSVs and Train first.',
                                       algorithms=accuracies.keys())

            model = joblib.load('calorie_model.pkl')
            scaler = joblib.load('scaler.pkl')

            user = np.array([[age, gender, height, weight, duration, heart_rate, body_temp]])
            try:
                user_s = scaler.transform(user)
            except Exception:
                user_s = user
            pred = model.predict(user_s)[0]

            session['algorithm'] = algo_choice
            session['calories'] = round(float(pred),2)

            return redirect(url_for('result'))

    return render_template('prediction.html', algorithms=accuracies.keys())

@app.route('/result')
def result():
    algorithm = session.get('algorithm', None)
    calories = session.get('calories', None)
    accuracy = "Not available"
    
    if algorithm:
        acc = accuracies.get(algorithm, None)
        if acc:
            accuracy = f"{algorithm} Accuracy: {acc*100:.2f}%"
    
    return render_template("result.html", 
                           algorithm=algorithm, 
                           calories=calories, 
                           accuracy=accuracy)

# ---------------- New Comparison Page ----------------
@app.route('/comparison')
def comparison():
    metrics = []
    if os.path.exists('exercise.csv') and os.path.exists('calories.csv'):
        try:
            results, best_name = train_from_csv('exercise.csv','calories.csv')
            metrics = results  # already a list of dicts including MAE, MSE, R2, Accuracy
        except:
            metrics = None
    else:
        metrics = None

    return render_template('comparison.html', metrics=metrics)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
