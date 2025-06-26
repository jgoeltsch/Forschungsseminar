import requests
import pandas as pd
from datetime import datetime

# Funktion: Wetterdaten (stündlich) von Open-Meteo abrufen
def get_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 51.751111,
        "longitude": 4.208611,
        "hourly": ["wind_speed_120m", "global_tilted_irradiance_instant"],
        "timezone": "Europe/Berlin",
        "tilt": 45
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Fehler beim Abrufen:", response.status_code)
        return None

    data = response.json()
    hourly = data['hourly']
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    return df