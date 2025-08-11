import pvlib
import math
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_energy(
        df, 
        solar_peak_power, num_modules, area_per_module, module_efficiency, tilt_angle, 
        wind_peak_power,r, h, turbine_efficiency, cut_in, cut_out, num_turbines, 
        yearly_demand, num_houses
        ):
    
    def solar_energy(solarradiation, cloudcover):
        tilt_angle_rad = math.radians(tilt_angle) # Neigungswinkel der Solarpanels [rad] (Annahme)

        f_dif = cloudcover / 100  # Diffuse Strahlung (0-1)
        f_dir = 1 - f_dif # Direkte Strahlung (0-1) basierend auf Bewölkung

        gti = solarradiation * (f_dir * math.cos(tilt_angle_rad) + f_dif) # Globalstrahlung auf die Solarpanels [W/m²]

        area = num_modules * area_per_module # Fläche der Solarpanels [m²]

        # Berechnung der Energieproduktion
        P_solar = module_efficiency * gti * area / 1_000_000 # Leistung [MW]
        
        # Begrenzung der Leistung auf die Nennleistung
        if P_solar > solar_peak_power:
            P_solar = solar_peak_power

        return P_solar

    def wind_energy(windspeed, sealevelpressure, temp, humidity):

        # Parameterübergabe 
        v = windspeed / 3.6 # Windgeschwindigkeit [m/s]
        p_L0 = sealevelpressure # Druck [hPa]
        T_L = temp # Temperatur Luft [°C]
        phi_rel = humidity # relative Luftfeuchtigkeit

        Rl = 287.05  # Gaskonstante trockene Luft [J/(kg·K)]
        Rd = 461.0   # Gaskonstante Wasserdampf [J/(kg·K)]

        # Cut-in / Cut-out Bereich
        if not (cut_in < v < cut_out):
            return 0
        
        # vereinfachte barometrische Höhenformel (Umrechnung Druch HNÜ auf Höhe)
        p_L90 = p_L0 * (1 - (0.0065 * h) / (T_L + 273.15)) ** 5.255 # Druck in hPa auf 120m Höhe

        # Sättigungsdampfdruck (Magnus-Formel), Ergebnis in Pascal
        pd = 611.213 * math.exp((17.5043 * T_L) / (241.2 + T_L))

        # Gaskonstante feuchte Luft Rf
        rf_denom = 1 - (phi_rel * (pd / (p_L90 * 100))) * (1 - Rl / Rd)
        Rf = Rl / rf_denom

        # Berechnung der Luftdichte
        rho_L = (p_L90 * 100) / (Rf * (T_L + 273.15))

        # Berechnung Leistung einer Windkraftanlage 
        P_wind = (rho_L * r**2 * math.pi * v**3)/2 # Leistung in Watt
        P_wind = turbine_efficiency * P_wind * num_turbines
        P_wind = P_wind / 1_000_000 # Leistung in MW

        # Begrenzung der Leistung auf die Nennleistung
        if P_wind > wind_peak_power:
            P_wind = wind_peak_power 
        
        return P_wind
        
    def demand_energy(timestamp):

        hourly_demand = yearly_demand * num_houses / (365 * 24)  # Stündlicher Gesamtverbrauch [MWh]

        hour = pd.to_datetime(timestamp).hour
        factor = (
            0.6 if hour < 6 or hour >= 22 else
            1.2 if hour < 9 else
            1.1 if hour < 17 else
            1.4
        )
        return hourly_demand * factor
    
    tilt_angle_rad = math.radians(tilt_angle)
    df['gti'] = df.apply(
        lambda row: row['solarradiation'] * ((1 - row['cloudcover'] / 100) * math.cos(tilt_angle_rad) + (row['cloudcover'] / 100)),
        axis=1
    )
    df['solar_energy_production'] = df.apply(
    lambda row: solar_energy(row['solarradiation'], row['cloudcover']),
    axis=1
    )
    df['wind_energy_production'] = df.apply(
        lambda row: wind_energy(row['windspeed'], row['sealevelpressure'], row['temp'], row['humidity']),
        axis=1
    )
    df['energy_demand'] = df['datetime'].apply(demand_energy)
    df['total_energy_production'] = df['solar_energy_production'] + df['wind_energy_production']
    
    return df[[
        'datetime', 'gti', 'windspeed',
        'solar_energy_production', 'wind_energy_production',
        'energy_demand', 'total_energy_production'
    ]]