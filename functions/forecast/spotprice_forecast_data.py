import pandas as pd


def get_spotprice_forecast():
    try:
        df = pd.read_csv(
            r"data/_validation/ds_forecast_spotprice_20_21Aug.csv",
            sep=",",
            dtype=str
        )[["Von", "Prognosepreis (ct/kWh)", "Tatsächlicher Preis (ct/kWh)"]]

        # Zeitstempel als datetime
        dt_local_naive = pd.to_datetime(
            df["Von"].str.strip(),
            dayfirst=True,
            errors="coerce"
        )

        forecast_price = pd.to_numeric(
            df["Prognosepreis (ct/kWh)"],
            errors="coerce"
        )
        real_price = pd.to_numeric(
            df["Tatsächlicher Preis (ct/kWh)"],
            errors="coerce"
        )

        out_forecast = (
            pd.DataFrame({
                "datetime": dt_local_naive,
                "spotprice": forecast_price,
                
            })
            .dropna(subset=["datetime", "spotprice"])
            .sort_values("datetime")
        )

        out_real = (
            pd.DataFrame({
                "datetime": dt_local_naive,
                "spotprice": real_price
            })
            .dropna(subset=["datetime", "spotprice"])
            .sort_values("datetime")
        )
        return out_forecast, out_real

    except Exception as e:
        print("Fehler beim Einlesen der Prognose-Spotpreise:", e)
        return None

#Quelle Datensatz: https://www.energyforecast.de/predictions_performance