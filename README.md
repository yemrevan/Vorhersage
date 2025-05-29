# Vorhersage – Train Delay Prediction

This repository includes the machine learning model training and prediction API for estimating train delays in Germany.

## Files

- `training2.py`: Trains the regression and classification models.
- `predict2.py`: Flask API that provides predictions via a `/predict` endpoint.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: For containerized deployment.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt

### Model Files

Trained models and encoders are stored externally due to file size limitations.
You can download them from the following link:
[Google Drive Link](https://drive.google.com/drive/folders/1g0-Avh2SPoP_rV5Ef9mGfeEP1D2xF7C9?usp=sharing)

These files include:

-model.pkl
-classifier.pkl
-encoders.pkl
-class_mapping.pkl

### Running the API
Start the prediction service:
python predict2.py

It will be available at: http://0.0.0.0:5002/predict

### Example Request (JSON)

[
  [
    {
      "station_name_from": "Berlin Hbf",
      "station_name_to": "Hamburg Hbf",
      "train_number": 123,
      "reihenfolge_from": 1,
      "planned_arrival_date_from": "2025-05-29",
      "planned_departure_from": "08:30",
      "planned_arrival_to": "10:00",
      "station_avg_delay_7_30": 5.2,
      "train_avg_delay_7_30": 3.7
    }
  ]
]

### Notes
-Delay is predicted both in minutes and as probability per category.
-Weather data and holidays are automatically considered if available.
















Sie können auf die .pkl-Dokumente über den Link zugreifen.

[https://drive.google.com/drive/u/0/folders/1g0-Avh2SPoP_rV5Ef9mGfeEP1D2xF7C9](https://drive.google.com/drive/folders/1g0-Avh2SPoP_rV5Ef9mGfeEP1D2xF7C9?usp=sharing)
