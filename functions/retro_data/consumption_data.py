import pandas as pd

def get_demand(num_houses, yearly_demand):
   

    df = pd.read_csv(r"data\ds_energyconsumption_24_25.csv", sep=";", dtype=str)

    # Unnötige Spalten entfernen
    drop_cols = [
        "Datum bis",
        "Netzlast inkl. Pumpspeicher [MWh] Berechnete Auflösungen",
        "Pumpspeicher [MWh] Berechnete Auflösungen",
        "Residuallast [MWh] Berechnete Auflösungen",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Zeitspalte parsen
    dt_local_naive = pd.to_datetime(df["Datum von"].str.strip(), dayfirst=True, errors="coerce")

    # Deutsches Zahlenformat -> Float
    demand = (
        df["Netzlast [MWh] Berechnete Auflösungen"]
        .str.replace(".", "", regex=False)   # Tausenderpunkt entfernen
        .str.replace(",", ".", regex=False)  # Dezimalkomma -> Punkt
    )
    demand = pd.to_numeric(demand, errors="coerce")

    out = (
        pd.DataFrame({"datetime": dt_local_naive, "energy_demand": demand})
        .dropna(subset=["datetime", "energy_demand"])
        .sort_values("datetime")
    )

    # Mittelwert berechnen und speichern
    hourly_demand_per_house = yearly_demand / (365*24)
    hourly_demand_total = hourly_demand_per_house * num_houses

    n_first_year = 365 * 24
    mean_demand = out["energy_demand"].iloc[:n_first_year].mean()
    scale_factor = mean_demand / hourly_demand_total
    
    out["energy_demand"] = out["energy_demand"] / scale_factor

    return out


#Quelle Datensatz: https://www.smard.de/home/downloadcenter/download-marktdaten/

