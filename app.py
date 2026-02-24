from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ==============================
# 1️⃣ LOAD DATASET
# ==============================

df = pd.read_csv("smart_home_energy_dataset_june_july.csv")

selected_users = ['User1', 'User2', 'User3']
df = df[df['user_id'].isin(selected_users)].reset_index(drop=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# ==============================
# 2️⃣ CREATE CLASSIFICATION LABEL
# ==============================

threshold = df['energy_consumed_kwh'].median()
df['usage_category'] = df['energy_consumed_kwh'].apply(
    lambda x: 'HIGH' if x >= threshold else 'LOW'
)

# ==============================
# 3️⃣ TRAIN DECISION TREE MODEL
# ==============================

features = [
    'device_power_rating',
    'tariff_rate',
    'duration_hours',
    'temperature',
    'hour'
]

X = df[features]
y = df['usage_category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# ==============================
# 4️⃣ PEAK HOURS CALCULATION
# ==============================

peak_hours_dict = {}
grouped = df.groupby(
    ['user_id', 'appliance_name', 'hour']
)['energy_consumed_kwh'].sum().reset_index()

for user in grouped['user_id'].unique():
    peak_hours_dict[user] = {}
    user_data = grouped[grouped['user_id'] == user]

    for appliance in user_data['appliance_name'].unique():
        data = user_data[user_data['appliance_name'] == appliance]
        max_energy = data['energy_consumed_kwh'].max()
        peak_hours = data[
            data['energy_consumed_kwh'] == max_energy
        ]['hour'].tolist()

        peak_hours_dict[user][appliance.lower()] = peak_hours

# ==============================
# 5️⃣ SUGGESTION FUNCTION
# ==============================

def get_tips(user_id, appliance, hour, tariff, prediction):
    appliance = appliance.lower()
    tips = []

    user_peaks = peak_hours_dict.get(user_id, {}).get(appliance, [])

    if tariff > 6:
        tips.append("⚠️ Current tariff is high.")
    else:
        tips.append("✅ Tariff is low.")

    if hour in user_peaks:
        tips.append("⚠️ You are using appliance during peak usage hours.")

    if prediction == "HIGH":
        tips.append("🔄 Consider shifting usage to off-peak hours.")

    return tips

# ==============================
# 6️⃣ ROUTES
# ==============================

@app.route("/")
def home():
    return "Energy Backend Running Successfully 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([{
        'device_power_rating': data['device_power_rating'],
        'tariff_rate': data['tariff_rate'],
        'duration_hours': data['duration_hours'],
        'temperature': data['temperature'],
        'hour': data['hour']
    }])

    prediction = dt_model.predict(input_df)[0]

    tips = get_tips(
        user_id=data['user_id'],
        appliance=data['appliance_name'],
        hour=data['hour'],
        tariff=data['tariff_rate'],
        prediction=prediction
    )

    return jsonify({
        "prediction": prediction,
        "tips": tips
    })

# ==============================
# 7️⃣ RUN SERVER (Render Compatible)
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
