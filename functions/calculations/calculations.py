import pandas as pd

def calculate_energy(df, solar_peak_power, wind_peak_power, hourly_demand):
    df = df.rename(columns={
        "time": "datetime",
        "global_tilted_irradiance_instant": "solarenergy",
        "wind_speed_120m": "windspeed"
    })

    def solar_energy(solarenergy):
        peak_power_m2 = 200
        area = solar_peak_power * 1_000_000  / peak_power_m2
        solar_power = solarenergy * 277.778 * area * 0.2 / 1000
        return solar_peak_power * (solar_power / peak_power_m2) if solar_power <= peak_power_m2 else solar_peak_power

    def wind_energy(windspeed):
        v = windspeed / 3.6
        if v < 2 or v > 25:
            return 0
        elif v <= 12:
            return wind_peak_power * ((v - 2) / 10)**3 * 0.5
        return wind_peak_power

    def demand_energy(timestamp):
        hour = pd.to_datetime(timestamp).hour
        factor = (
            0.6 if hour < 6 or hour >= 22 else
            1.2 if hour < 9 else
            1.1 if hour < 17 else
            1.4
        )
        return hourly_demand * factor

    df['solar_energy_production'] = df['solarenergy'].apply(solar_energy)
    df['wind_energy_production'] = df['windspeed'].apply(wind_energy)
    df['energy_demand'] = df['datetime'].apply(demand_energy)
    df['total_energy_production'] = df['solar_energy_production'] + df['wind_energy_production']
    return df