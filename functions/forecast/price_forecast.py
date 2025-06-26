import requests
import pandas as pd
import pytz
import numpy as np
from datetime import datetime

def price_forecast():
    url = "https://www.energyforecast.de/api/v1/predictions/next_48_hours"
    params = {
        'token': "ab91918588709b1a60ac9993eb",
        'fixed_cost_cent': 0,
        'vat': 0
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Fehler beim Abruf:", response.status_code)
        return None

    data = response.json()
    records = [
        {
            'datetime': entry['start'],
            'price': round(entry['price'] * 100, 2),
        }
        for entry in data
    ]

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)


    # Aktuelle Zeit (Berlin) für Filter
    now = datetime.now(pytz.timezone('Europe/Berlin')).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    df = df[df['datetime'] >= now].reset_index(drop=True)

    return df

# Funktion Verbrauchsrechner
def parse_time_to_hours(time_str):
    try:
        h, m = map(int, time_str.split(":"))
        return h + m / 60
    except:
        return 0.0