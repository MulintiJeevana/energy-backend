# =============================
# Complete Smart Home Energy Analysis & Prediction
# =============================

# ----------------------------
# Step 0: Install dependencies
# ----------------------------
!pip install openpyxl

# ----------------------------
# Step 1: Upload dataset
# ----------------------------
from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file (update filename if different)
df = pd.read_csv(list(uploaded.keys())[0])

# ----------------------------
# Step 2: Filter selected users
# ----------------------------
selected_users = ['User1', 'User2', 'User3']
df = df[df['user_id'].isin(selected_users)].reset_index(drop=True)

# ----------------------------
# Step 3: Preprocessing
# ----------------------------
# Convert timestamp and extract hour, day, month, weekday
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

# Drop rows with missing values
df = df.dropna()

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=['appliance_name', 'room'], drop_first=True)

# ----------------------------
# Step 4: Regression: Predict energy_consumed_kwh
# ----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sort dataset by timestamp
df_encoded_sorted = df_encoded.sort_values(by='timestamp').reset_index(drop=True)
X_sorted = df_encoded_sorted.drop(['energy_consumed_kwh', 'timestamp'], axis=1)
y_sorted = df_encoded_sorted['energy_consumed_kwh']

# One-hot encode remaining categorical columns if any
categorical_cols = X_sorted.select_dtypes(include='object').columns.tolist()
X_encoded_sorted = pd.get_dummies(X_sorted, columns=categorical_cols, drop_first=True)

# Time-based split (80% train, 20% test)
split_index = int(len(X_encoded_sorted) * 0.8)
X_train = X_encoded_sorted.iloc[:split_index]
X_test = X_encoded_sorted.iloc[split_index:]
y_train = y_sorted.iloc[:split_index]
y_test = y_sorted.iloc[split_index:]

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Regression: MSE =", mse, ", R2 =", r2)

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Energy Consumed (kWh)")
plt.ylabel("Predicted Energy Consumed (kWh)")
plt.title("Linear Regression: Predictions vs Actual")
plt.grid(True)
plt.show()

# ----------------------------
# Step 5: Classification: Label usage as HIGH/LOW
# ----------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create usage_category based on median
threshold = df_encoded['energy_consumed_kwh'].median()
df_encoded['usage_category'] = df_encoded['energy_consumed_kwh'].apply(
    lambda x: 'HIGH' if x >= threshold else 'LOW'
)

# Select features for classification
selected_features = ['device_power_rating', 'tariff_rate', 'duration_hours', 'temperature', 'hour']
X_cls = df_encoded[selected_features]
y_cls = df_encoded['usage_category']

# Train-test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Train Decision Tree classifier
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_cls, y_train_cls)

# Evaluate classifier
y_pred_cls = dt_model.predict(X_test_cls)
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_cls))
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls))
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))

# Feature importance
importances = dt_model.feature_importances_
features = X_cls.columns
indices = importances.argsort()[::-1]
plt.figure(figsize=(8, 5))
plt.title("Feature Importance - Decision Tree")
plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ----------------------------
# Step 6: Peak hours calculation
# ----------------------------
def hour_to_ampm(hour):
    if hour == 0: return "12 AM"
    elif hour == 12: return "12 PM"
    elif hour < 12: return f"{hour} AM"
    else: return f"{hour-12} PM"

def hours_to_intervals(hours_list):
    if not hours_list: return "No data"
    hours_sorted = sorted(hours_list)
    intervals = []
    start = end = hours_sorted[0]
    for h in hours_sorted[1:]:
        if h == end + 1: end = h
        else:
            if start == end: intervals.append(hour_to_ampm(start))
            else: intervals.append(f"{hour_to_ampm(start)}-{hour_to_ampm(end)}")
            start = end = h
    if start == end: intervals.append(hour_to_ampm(start))
    else: intervals.append(f"{hour_to_ampm(start)}-{hour_to_ampm(end)}")
    return ", ".join(intervals)

peak_hours_dict = {}
user_peak_hours = df.groupby(['user_id', 'appliance_name', df['timestamp'].dt.hour])['energy_consumed_kwh'].sum().reset_index()
user_peak_hours.rename(columns={'timestamp':'hour'}, inplace=True)
for user in user_peak_hours['user_id'].unique():
    peak_hours_dict[user] = {}
    user_data = user_peak_hours[user_peak_hours['user_id']==user]
    for appliance in user_data['appliance_name'].unique():
        appliance_data = user_data[user_data['appliance_name']==appliance]
        max_energy = appliance_data['energy_consumed_kwh'].max()
        peak_hours = appliance_data[appliance_data['energy_consumed_kwh']==max_energy]['hour'].tolist()
        if not peak_hours: peak_hours = [appliance_data['hour'].iloc[0]]
        peak_hours_dict[user][appliance.lower()] = {
            "hours_list": peak_hours,
            "interval_str": hours_to_intervals(peak_hours)
        }

# ----------------------------
# Step 7: Off-peak hours calculation
# ----------------------------
appliance_hourly = df.groupby(['appliance_name', df['timestamp'].dt.hour])['energy_consumed_kwh'].sum().reset_index()
appliance_hourly.rename(columns={'timestamp':'hour'}, inplace=True)
off_peak_hours_dict = {}
for appliance in appliance_hourly['appliance_name'].unique():
    data = appliance_hourly[appliance_hourly['appliance_name']==appliance]
    data_sorted = data.sort_values(by='energy_consumed_kwh', ascending=True)
    threshold = max(1, int(len(data_sorted)*0.25))
    bottom_hours = data_sorted.head(threshold)['hour'].tolist()
    if not bottom_hours: bottom_hours = [data_sorted['hour'].min()]
    off_peak_hours_dict[appliance.lower()] = {
        "hours_list": bottom_hours,
        "interval_str": hours_to_intervals(bottom_hours)
    }

# ----------------------------
# Step 8: Predict and generate suggestions (requires predict_and_suggest function)
# ----------------------------
# Keep original df for appliance_name and timestamp
df_original = df.copy()
features = ['device_power_rating', 'tariff_rate', 'duration_hours', 'temperature',]()
