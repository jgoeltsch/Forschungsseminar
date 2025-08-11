import pandas as pd

def get_spotprice():
    try:
        df = pd.read_csv(
            r"data\ds_spotprice_EPEX_SPOT_24_25.csv",
            sep=";",
            dtype=str
        )[["Datum", "von", "Spotmarktpreis in ct/kWh"]]

        # Zeitstempel: Europe/Berlin -> naive lokale Zeit
        dt_local_naive = pd.to_datetime(
            df["Datum"].str.strip() + " " + df["von"].str.strip(),
            dayfirst=True,
            errors="coerce"
        ).dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        ).dt.tz_localize(None)  # Zeitzone entfernen, lokale Uhrzeit behalten

        # Preis als float
        price = pd.to_numeric(
            df["Spotmarktpreis in ct/kWh"].str.replace(",", ".", regex=False),
            errors="coerce"
        )

        # DataFrame
        out = (
            pd.DataFrame({"datetime": dt_local_naive, "spotprice": price})
            .dropna(subset=["datetime", "spotprice"])
            .sort_values("datetime")
        )

        # Letzte 7 Tage
        last_time = out["datetime"].max()
        start_time = last_time - pd.Timedelta(days=7)
        out_recent = out[out["datetime"] >= start_time].reset_index(drop=True)

        return out_recent

    except Exception as e:
        print("Fehler beim Einlesen der Spotpreise:", e)
        return None
