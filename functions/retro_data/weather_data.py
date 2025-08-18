import pandas as pd

def get_weather():
    try:
        df = pd.read_csv("data\ds_weather_Harinvliet_Zuid_24_25.csv", parse_dates=["datetime"])
        df = df.sort_values("datetime")
        df = df[["datetime", "solarradiation", "windspeed", "temp", "cloudcover", "humidity", "sealevelpressure"]].reset_index(drop=True)

        return df

    except Exception as e:
        print("Fehler beim Einlesen der Wetterdaten:", e)
        return None

#Quelle Datensatz: https://www.visualcrossing.com/weather-query-builder/