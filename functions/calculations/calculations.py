import pvlib
import math
import pandas as pd
import numpy as np

def calculate_energy(
        df, 
        solar_peak_power, num_modules, area_per_module, module_efficiency, tilt_angle, latitude, longitude, azimuth, albedo, 
        wind_peak_power, r, h, turbine_efficiency, cut_in, cut_out, num_turbines, 
        yearly_demand, num_houses,
    ):

    def solar_energy(datetime, solarradiation, cloudcover):
        
        # Sonnenstand berechnen
        solpos = pvlib.solarposition.get_solarposition(datetime, latitude, longitude)
        solar_elevation = solpos['elevation'].values[0]
        solar_zenith = solpos['zenith'].values[0]
        solar_azimuth = solpos['azimuth'].values[0]

        # Aufteilung der GHI in DHI und DNI (basierend auf Cloud Cover)
        ghi = solarradiation
        f_dif = cloudcover / 100
        dhi = f_dif * ghi
        dni = (ghi - dhi) / max(np.sin(np.radians(solar_elevation)), 1e-6)

        # POA-Strahlung berechnen
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt_angle,
            surface_azimuth=azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            albedo=albedo
        )

        poa_irradiance = poa['poa_global']  # Einstrahlung auf Panel [W/m²]

        area = num_modules * area_per_module
        P_solar = module_efficiency * poa_irradiance * area / 1_000_000  # [MW]

        return min(P_solar, solar_peak_power)

    def wind_energy(windspeed, sealevelpressure, temp, humidity):
        v = windspeed / 3.6
        p_L0 = sealevelpressure
        T_L = temp
        phi_rel = humidity

        Rl = 287.05
        Rd = 461.0

        if not (cut_in < v < cut_out):
            return 0

        p_L90 = p_L0 * (1 - (0.0065 * h) / (T_L + 273.15)) ** 5.255
        pd = 611.213 * math.exp((17.5043 * T_L) / (241.2 + T_L))
        rf_denom = 1 - (phi_rel * (pd / (p_L90 * 100))) * (1 - Rl / Rd)
        Rf = Rl / rf_denom
        rho_L = (p_L90 * 100) / (Rf * (T_L + 273.15))

        P_wind = (rho_L * r**2 * math.pi * v**3) / 2
        P_wind = turbine_efficiency * P_wind * num_turbines
        P_wind = P_wind / 1_000_000  # [MW]

        return min(P_wind, wind_peak_power)

    def demand_energy(timestamp):
        hourly_demand = yearly_demand * num_houses / (365 * 24)
        hour = timestamp.hour
        factor = (
            0.6 if hour < 6 or hour >= 22 else
            1.2 if hour < 9 else
            1.1 if hour < 17 else
            1.4
        )
        return hourly_demand * factor

    # Berechnungen auf DataFrame anwenden
    df['solar_energy_production'] = df.apply(
        lambda row: solar_energy(row['datetime'], row['solarradiation'], row['cloudcover']),
        axis=1
    )
    df['wind_energy_production'] = df.apply(
        lambda row: wind_energy(row['windspeed'], row['sealevelpressure'], row['temp'], row['humidity']),
        axis=1
    )
    df['energy_demand'] = df['datetime'].apply(demand_energy)
    df['total_energy_production'] = df['solar_energy_production'] + df['wind_energy_production']

    df['poa_irradiance'] = df.apply(
    lambda row: pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_angle,
        surface_azimuth=azimuth,
        solar_zenith=pvlib.solarposition.get_solarposition(row['datetime'], latitude, longitude)['zenith'].values[0],
        solar_azimuth=pvlib.solarposition.get_solarposition(row['datetime'], latitude, longitude)['azimuth'].values[0],
        dni=(row['solarradiation'] - row['cloudcover'] / 100 * row['solarradiation']) / max(np.sin(np.radians(
            pvlib.solarposition.get_solarposition(row['datetime'], latitude, longitude)['elevation'].values[0])), 1e-6),
        ghi=row['solarradiation'],
        dhi=row['cloudcover'] / 100 * row['solarradiation'],
        albedo=albedo
    )['poa_global'],
    axis=1
)

    return df[[
        'datetime', 'windspeed', 'poa_irradiance',
        'solar_energy_production', 'wind_energy_production',
        'energy_demand', 'total_energy_production'
    ]]
