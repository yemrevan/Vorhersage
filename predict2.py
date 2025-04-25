#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import json
import sys
from datetime import datetime, timedelta

# Modelleri ve encoder'ları yükle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Feature sırasını belirle (Eğitimdeki sırayla aynı)
feature_cols = [
    "station_name_from",
    "station_name_to",
    "train_number",
    "day_of_week",
    "planned_hour_from",
    "soll_dauer",
    "reihenfolge_from",
    "station_avg_delay_7_30",
    "train_avg_delay_7_30"
]

# API'den gelen JSON inputu al
input_json = sys.argv[1]  # JSON string
input_data = json.loads(input_json)

results = []

for item in input_data:
    # departure_date + planned_departure_from birleşimi → datetime
    departure_dt = datetime.strptime(item['departure_date'] + " " + item['planned_departure_from'], "%Y-%m-%d %H:%M")
    arrival_time = datetime.strptime(item['planned_arrival_to'], "%H:%M").time()

    # arrival_time'ı departure_date ile birleştir (ertesi gün olabilir!)
    arrival_dt = datetime.combine(departure_dt.date(), arrival_time)
    if arrival_dt < departure_dt:
        arrival_dt += timedelta(days=1)  # Ertesi günse bir gün ekle

    # Gün, saat, süre hesapla
    day_of_week = departure_dt.weekday()
    planned_hour = departure_dt.hour
    soll_dauer = (arrival_dt - departure_dt).total_seconds() / 60  # dakika

    # Label encoding (station names)
    station_from_enc = encoders['le_from'].transform([item['station_from']])[0]
    station_to_enc = encoders['le_to'].transform([item['station_to']])[0]

    # Eksik feature'ları input'tan al
    reihenfolge_from = item['reihenfolge_from']
    station_avg_delay_7_30 = item['station_avg_delay_7_30']
    train_avg_delay_7_30 = item['train_avg_delay_7_30']

    # Feature dataframe (doğru sıralama!)
    input_df = pd.DataFrame([{
        'station_name_from': station_from_enc,
        'station_name_to': station_to_enc,
        'train_number': item['train_number'],
        'day_of_week': day_of_week,
        'planned_hour_from': planned_hour,
        'soll_dauer': soll_dauer,
        'reihenfolge_from': reihenfolge_from,
        'station_avg_delay_7_30': station_avg_delay_7_30,
        'train_avg_delay_7_30': train_avg_delay_7_30
    }])[feature_cols]  # Sıralama burada netleşiyor

    # Regresyon tahmini (dakika)
    predicted_delay = model.predict(input_df)[0]

    # Classification tahmini (olasılık yüzdeleri)
    probabilities = classifier.predict_proba(input_df)[0]
    category_probs = {
        class_mapping[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)
    }

    # Sonuçları birleştir
    results.append({
        "station_from": item['station_from'],
        "station_to": item['station_to'],
        "train_number": item['train_number'],
        "departure_date": item['departure_date'],
        "planned_departure_from": item['planned_departure_from'],
        "planned_arrival_to": item['planned_arrival_to'],
        "predicted_delay": round(predicted_delay, 2),
        "category_probabilities": category_probs
    })

# Sonuçları JSON olarak yazdır
print(json.dumps(results, indent=2))


# In[ ]:




