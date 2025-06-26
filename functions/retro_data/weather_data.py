import pandas as pd

def get_weather():
    try:
        df = pd.read_csv("data\ds_weather_Harinvliet_Zuid_24_25.csv", parse_dates=["datetime"])
        df = df.sort_values("datetime")

        last_time = df["datetime"].max()
        start_time = last_time - pd.Timedelta(days=7)

        df_recent = df[df["datetime"] >= start_time].copy()
        df_recent = df_recent[["datetime", "solarenergy", "windspeed"]].reset_index(drop=True)

        return df_recent

    except Exception as e:
        print("Fehler beim Einlesen der Wetterdaten:", e)
        return None