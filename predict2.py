from flask import Flask, request, jsonify
import pickle
from datetime import datetime, timedelta
import sqlalchemy
import pandas as pd
import holidays

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

db_user = 'postgres'
db_password = 'UWP12345!'
db_host = '35.246.149.161'
db_port = '5432'
db_name = 'postgres'
connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = sqlalchemy.create_engine(connection_string)


de_holidays = holidays.Germany(years=2025)


feature_cols = [
    "station_name_from",
    "station_name_to",
    "train_number",
    "day_of_week",
    "planned_hour_from",
    "soll_dauer",
    "reihenfolge_from",
    "station_avg_delay_7_30",
    "train_avg_delay_7_30",
    "day_period",
    "is_holiday",
    "is_peak_time",
    "is_weekend",
    "wetter"
]


def map_day_period(hour):
    if 0 <= hour < 6:
        return "0-6"
    elif 6 <= hour < 12:
        return "6-12"
    elif 12 <= hour < 18:
        return "12-18"
    else:
        return "18-0"


def get_weather_data(station_date_pairs):
    try:
        # Benzersiz (station_name, date) çiftlerini al
        unique_pairs = list(set((station, date) for station, date in station_date_pairs))
        if not unique_pairs:
            return {}

        
        query = """
            SELECT station_name, date, weather
            FROM wetterdaten_zukunft
            WHERE (station_name, date) IN %s
        """

        params = [tuple(pair) for pair in unique_pairs]
        result = pd.read_sql_query(query, engine, params=(tuple(params),))
        

        weather_dict = {(row["station_name"], str(row["date"])): row["weather"] for _, row in result.iterrows()}
        return weather_dict
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        results = []
        
        
        station_date_pairs = []
        for route in input_data:
            for item in route:
                station_date_pairs.append((item['station_name_from'], item['planned_arrival_date_from']))

        
        weather_dict = get_weather_data(station_date_pairs)

        for route in input_data:
            for item in route:
                # Tarih ve saat formatını işle
                try:
                    # GMT
                    try:
                        departure_dt = datetime.strptime(
                            item['planned_arrival_date_from'] + " " + item['planned_departure_from'],
                            "%a, %d %b %Y %H:%M:%S GMT"
                        )
                    except ValueError:
                       
                        try:
                            departure_dt = datetime.strptime(
                                item['planned_arrival_date_from'] + " " + item['planned_departure_from'],
                                "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            departure_dt = datetime.strptime(
                                item['planned_arrival_date_from'] + " " + item['planned_departure_from'],
                                "%Y-%m-%d %H:%M"
                            )
                except ValueError:
                    return jsonify({"error": "Invalid departure time format. Expected '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', or '%a, %d %b %Y %H:%M:%S GMT'"}), 400

                try:
                    # Önce GMT formatını dene
                    try:
                        arrival_time = datetime.strptime(item['planned_arrival_to'], "%a, %d %b %Y %H:%M:%S GMT").time()
                    except ValueError:
                        
                        try:
                            arrival_time = datetime.strptime(item['planned_arrival_to'], "%H:%M:%S").time()
                        except ValueError:
                            arrival_time = datetime.strptime(item['planned_arrival_to'], "%H:%M").time()
                except ValueError:
                    return jsonify({"error": "Invalid arrival time format. Expected '%H:%M:%S', '%H:%M', or '%a, %d %b %Y %H:%M:%S GMT'"}), 400

                
                arrival_dt = datetime.combine(departure_dt.date(), arrival_time)
                if arrival_dt < departure_dt:
                    arrival_dt += timedelta(days=1)

                
                day_of_week = departure_dt.weekday()
                planned_hour = departure_dt.hour
                soll_dauer = (arrival_dt - departure_dt).total_seconds() / 60
                is_peak_time = 1 if 7 <= planned_hour <= 9 or 16 <= planned_hour <= 19 else 0
                day_period = map_day_period(planned_hour)
                is_weekend = 1 if day_of_week in [5, 6] else 0
                is_holiday = 1 if datetime.strptime(item['planned_arrival_date_from'], "%Y-%m-%d").date() in de_holidays else 0
                wetter = weather_dict.get((item['station_name_from'], item['planned_arrival_date_from']), "unknown")

                # Label encoding
                try:
                    station_from_enc = encoders['le_from'].transform([item['station_name_from']])[0]
                    station_to_enc = encoders['le_to'].transform([item['station_name_to']])[0]
                    day_period_enc = encoders['le_day_period'].transform([day_period])[0]
                    wetter_enc = encoders['le_wetter'].transform([wetter])[0]
                except KeyError as e:
                    return jsonify({"error": f"Encoder missing: {str(e)}"}), 500
                except ValueError as e:
                    return jsonify({"error": f"Unknown value in encoding: {str(e)}"}), 400

                
                required_fields = ['reihenfolge_from', 'station_avg_delay_7_30', 'train_avg_delay_7_30']
                for field in required_fields:
                    if field not in item:
                        return jsonify({"error": f"Missing required field: {field}"}), 400

                
                input_features = [
                    station_from_enc,
                    station_to_enc,
                    item['train_number'],
                    day_of_week,
                    planned_hour,
                    soll_dauer,
                    item['reihenfolge_from'],
                    item['station_avg_delay_7_30'],
                    item['train_avg_delay_7_30'],
                    day_period_enc,
                    is_holiday,
                    is_peak_time,
                    is_weekend,
                    wetter_enc
                ]

                
                print(f"Input features: {dict(zip(feature_cols, input_features))}")
                print(f"day_period: {day_period}, encoded: {day_period_enc}")
                print(f"wetter: {wetter}, encoded: {wetter_enc}")

                
                try:
                    predicted_delay = model.predict([input_features])[0]
                    probabilities = classifier.predict_proba([input_features])[0]
                except ValueError as e:
                    return jsonify({"error": f"Prediction error: {str(e)}"}), 500

                category_probs = {
                    class_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)
                }

                
                results.append({
                    "station_name_from": item['station_name_from'],
                    "station_name_to": item['station_name_to'],
                    "train_number": item['train_number'],
                    "planned_arrival_date_from": item['planned_arrival_date_from'],
                    "planned_departure_from": item['planned_departure_from'],
                    "planned_arrival_to": item['planned_arrival_to'],
                    "predicted_delay": round(predicted_delay, 2),
                    "category_probabilities": category_probs
                })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
