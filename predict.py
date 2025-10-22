
import os
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import requests
import datetime
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 1. Load Random Forest model
# ----------------------------
rf_model_path =  os.path.join(os.path.dirname(__file__), 'models', 'random_forest_model .pkl')

with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

# ----------------------------
# 2. Load LSTM models + scalers
# ----------------------------
lstm_models = {}
scalers = {}

village_names = [
    "Khatima", "Kashipur", "Jaspur", "dineshpur", "Haldwani",
    "Nainital", "sitarganj", "nanakmatta", "Bazpur", "pantnagar", "Pilibhit"
]

for v in village_names:
    try:
        model = tf.keras.models.load_model(f'lstm_model_{v}.h5', compile=False)
        lstm_models[v] = model
        with open(f'scaler_{v}.pkl', 'rb') as f:
            scalers[v] = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load model/scaler for {v}: {e}")

# ----------------------------
# 3. Village coordinates and elevations
# ----------------------------
village_coords = {
    "Khatima": (28.92, 79.97), "Kashipur": (29.22, 78.96),
    "Jaspur": (29.29, 78.82), "dineshpur": (28.97, 79.42),
    "Haldwani": (29.22, 79.52), "Nainital": (29.38, 79.45),
    "sitarganj": (28.93, 79.70), "nanakmatta": (28.99, 79.88),
    "Bazpur": (29.15, 79.13), "pantnagar": (28.98, 79.41),
    "Pilibhit": (28.63, 79.80)
}

village_elevations = {
    "Khatima": 260, "Kashipur": 245, "Jaspur": 215, "dineshpur": 235,
    "Haldwani": 424, "Nainital": 1938, "sitarganj": 259, "nanakmatta": 230,
    "Bazpur": 205, "pantnagar": 243, "Pilibhit": 172
}

# ----------------------------
# 4. Fetch rainfall data
# ----------------------------
def fetch_recent_data(lat, lon, start_date, end_date):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}"
        "&parameters=PRECTOTCORR&community=AG&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json().get('properties', {}).get('parameter', {}).get('PRECTOTCORR', {})
    if not data:
        raise ValueError("No rainfall data found for coordinates")
    dates, values = [], []
    for dstr, val in data.items():
        try:
            dt = pd.to_datetime(dstr, format='%Y%m%d').date()
            dates.append(dt)
            values.append(float(val))
        except:
            continue
    df = pd.DataFrame({'date': dates, 'rainfall': values}).sort_values('date').reset_index(drop=True)
    df['rainfall'] = df['rainfall'].apply(lambda x: np.nan if x < 0 or x > 1000 else x).ffill().fillna(0)
    return df

# ----------------------------
# 5. Predict rainfall using LSTM
# ----------------------------
def predict_rainfall(village, last_sequence, days=7):
    lstm_model = lstm_models[village]
    scaler = scalers[village]
    current_seq = last_sequence.copy()
    future_scaled = []
    for _ in range(days):
        pred_scaled = lstm_model.predict(current_seq, verbose=0)[0, 0]
        future_scaled.append(pred_scaled)
        current_seq = np.concatenate((current_seq[:, 1:, :], np.array(pred_scaled).reshape(1,1,1)), axis=1)
    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_rain = scaler.inverse_transform(future_scaled).flatten()
    future_rain[future_rain < 0] = 0
    return future_rain

# ----------------------------
# 6. Predict flood, rainfall, elevation
# ----------------------------
def predict_flood(village, start_date=None, end_date=None):
    if village not in village_coords:
        raise ValueError(f"Village '{village}' not recognized")
    if village not in lstm_models or village not in scalers:
        raise ValueError(f"Model or scaler for {village} not loaded")

    lat, lon = village_coords[village]
    time_steps = 50

    # Determine prediction days
    if start_date and end_date:
        start_date_obj = pd.to_datetime(start_date).date()
        end_date_obj = pd.to_datetime(end_date).date()
        pred_days = (end_date_obj - start_date_obj).days + 1
        if pred_days <= 0:
            raise ValueError("End date must be after start date")
        fetch_days = time_steps + pred_days
        df = fetch_recent_data(lat, lon, start_date=start_date_obj - datetime.timedelta(days=time_steps-1), end_date=end_date_obj)
    else:
        # Future prediction: next 7 days
        pred_days = 7
        fetch_days = max(time_steps, 60)
        df = fetch_recent_data(lat, lon, start_date=datetime.date.today() - datetime.timedelta(days=fetch_days-1),
                               end_date=datetime.date.today())

    if len(df) < time_steps:
        raise ValueError("Not enough data to predict")

    # Prepare sequence
    scaler = scalers[village]
    rain_scaled = scaler.transform(df[['rainfall']].values)
    last_seq = rain_scaled[-time_steps:].reshape(1, time_steps, 1)
    future_rain = predict_rainfall(village, last_seq, days=pred_days)

    # Generate forecast dates
    last_date = df['date'].max()
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(pred_days)]

    # Random Forest flood prediction
    flood_preds = []
    for rain_val in future_rain:
        rf_input = np.array([[rain_val]])
        flood_preds.append(str(rf_model.predict(rf_input)[0]))

    elevation = village_elevations.get(village, None)
    forecast = []
    for d, r, f in zip(future_dates, future_rain, flood_preds):
        forecast.append({
            "date": d.strftime('%Y-%m-%d'),
            "rainfall": round(float(r), 2),
            "flood_risk": f,
            "elevation": elevation
        })

    return {"village": village, "forecast": forecast}
