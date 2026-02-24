# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('smart_home_energy_dataset_june_july.csv')

# Keep selected users
selected_users = ['User1', 'User2', 'User3']
df = df[df['user_id'].isin(selected_users)].reset_index(drop=True)

# Preprocess timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Label energy usage as HIGH or LOW
threshold = df_encoded['energy_consumed_kwh'].median()
df_encoded['usage_category'] = df_encoded['energy_consumed_kwh'].apply(
    lambda x: 'HIGH' if x >= threshold else 'LOW'
)

# Features for Decision Tree
features_cls = ['device_power_rating', 'tariff_rate', 'duration_hours', 'temperature', 'hour']
X_cls = df_encoded[features_cls]
y_cls = df_encoded['usage_category']

# Train-test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_cls, y_train_cls)

# -----------------------------
# Prepare peak and off-peak dictionaries
# -----------------------------
def hour_to_ampm(hour):
    if hour == 0:
        return "12 AM"
    elif hour == 12:
        return "12 PM"
    elif hour < 12:
        return f"{hour} AM"
    else:
        return f"{hour-12} PM"

def hours_to_intervals(hours_list):
    if not hours_list:
        return "No data"
    hours_sorted = sorted(hours_list)
    intervals = []
    start = hours_sorted[0]
    end = hours_sorted[0]
    for h in hours_sorted[1:]:
        if h == end + 1:
            end = h
        else:
            intervals.append(hour_to_ampm(start) if start == end else f"{hour_to_ampm(start)}-{hour_to_ampm(end)}")
            start = end = h
    intervals.append(hour_to_ampm(start) if start == end else f"{hour_to_ampm(start)}-{hour_to_ampm(end)}")
    return ", ".join(intervals)

# Peak hours per user-appliance
peak_hours_dict = {}
user_peak_hours = df.groupby(['user_id', 'appliance_name', df['timestamp'].dt.hour])['energy_consumed_kwh'].sum().reset_index()
user_peak_hours.rename(columns={'timestamp': 'hour'}, inplace=True)

for user in user_peak_hours['user_id'].unique():
    peak_hours_dict[user] = {}
    user_data = user_peak_hours[user_peak_hours['user_id'] == user]
    for appliance in user_data['appliance_name'].unique():
        appliance_data = user_data[user_data['appliance_name'] == appliance]
        max_energy = appliance_data['energy_consumed_kwh'].max()
        peak_hours = appliance_data[appliance_data['energy_consumed_kwh'] == max_energy]['hour'].tolist()
        if not peak_hours:
            peak_hours = [appliance_data['hour'].iloc[0]]
        peak_hours_dict[user][appliance.lower()] = {
            "hours_list": peak_hours,
            "interval_str": hours_to_intervals(peak_hours)
        }

# Off-peak hours per appliance
off_peak_hours_dict = {}
appliance_hourly = df.groupby(['appliance_name', df['timestamp'].dt.hour])['energy_consumed_kwh'].sum().reset_index()
appliance_hourly.rename(columns={'timestamp': 'hour'}, inplace=True)

for appliance in appliance_hourly['appliance_name'].unique():
    data = appliance_hourly[appliance_hourly['appliance_name'] == appliance]
    data_sorted = data.sort_values(by='energy_consumed_kwh', ascending=True)
    threshold_val = max(1, int(len(data_sorted) * 0.25))
    bottom_hours = data_sorted.head(threshold_val)['hour'].tolist()
    if not bottom_hours:
        bottom_hours = [data_sorted['hour'].min()]
    off_peak_hours_dict[appliance.lower()] = {
        "hours_list": bottom_hours,
        "interval_str": hours_to_intervals(bottom_hours)
    }

# -----------------------------
# Function to get suggestions
# -----------------------------
def get_optimized_energy_tips(user_id, appliance, hour, tariff, duration, model_prediction,
                              peak_hours_dict, off_peak_hours_dict):
    tips = []
    appliance = appliance.lower()
    user_peak_data = peak_hours_dict.get(user_id, {}).get(appliance, {"hours_list": [], "interval_str": ""})
    appliance_off_peak_data = off_peak_hours_dict.get(appliance, {"hours_list": [], "interval_str": ""})
    user_peak_hours_list = user_peak_data["hours_list"]
    user_peak_intervals_str = user_peak_data["interval_str"]
    appliance_off_peak_intervals_str = appliance_off_peak_data["interval_str"]
    show_off_peak = (model_prediction == "HIGH")

    if appliance in ["ac", "heater", "washing machine", "refrigerator"]:
        if tariff > 6:
            tips.append("⚠️ Current tariff is high — you are in peak hours.")
        else:
            tips.append("✅ Tariff is low — off-peak hours, optimal for appliance usage.")

    if appliance == "ac":
        if show_off_peak:
            tips.append(f"✅ Recommended off-peak hours for AC: {appliance_off_peak_intervals_str}")
        if hour in user_peak_hours_list:
            tips.append(f"⚠️ AC usage now is during your peak hours: {user_peak_intervals_str}")
    elif appliance == "heater":
        if show_off_peak:
            tips.append(f"✅ Recommended off-peak hours for Heater: {appliance_off_peak_intervals_str}")
    elif appliance == "washing machine":
        if show_off_peak:
            tips.append(f"✅ Recommended off-peak hours for Washing Machine: {appliance_off_peak_intervals_str}")
        if duration > 1.5:
            tips.append("🧺 Use quick or eco-friendly cycles to save water and electricity.")
    elif appliance == "fan":
        if hour in user_peak_hours_list:
            tips.append(f"⚠️ Using fan during peak hours: {user_peak_intervals_str}")
    elif appliance == "light":
        if hour in user_peak_hours_list:
            tips.append(f"⚠️ Using lights during peak hours: {user_peak_intervals_str}")
    elif appliance == "refrigerator":
        if show_off_peak:
            tips.append(f"✅ Recommended off-peak hours for Refrigerator: {appliance_off_peak_intervals_str}")
    else:
        tips.append(f"❓ No predefined tips for {appliance}.")

    return tips

def predict_and_suggest(input_data):
    df_pred = pd.DataFrame([{
        'device_power_rating': input_data['device_power_rating'],
        'tariff_rate': input_data['tariff_rate'],
        'duration_hours': input_data['duration_hours'],
        'temperature': input_data['temperature'],
        'hour': input_data['hour']
    }])
    prediction = dt_model.predict(df_pred)[0]
    tips = get_optimized_energy_tips(
        user_id=input_data['user_id'],
        appliance=input_data['appliance_name'],
        hour=input_data['hour'],
        tariff=input_data['tariff_rate'],
        duration=input_data['duration_hours'],
        model_prediction=prediction,
        peak_hours_dict=peak_hours_dict,
        off_peak_hours_dict=off_peak_hours_dict
    )
    return {"prediction": prediction, "tips": tips}

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Smart Home Energy Backend is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    result = predict_and_suggest(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
