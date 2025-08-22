def get_results(result_df_rule, result_df_opt, result_df_opt_fore, result_kpis_rule, result_kpis_opt, result_kpis_opt_fore, export_factor):
    
    # KPIs ausgeben
    print("Regelbasiert:")
    print(f"Netto Stromkosten: {result_kpis_rule['Netto Stromkosten']:.2f} €")
    print(f"Netzstromkosten: {result_kpis_rule['Netzstromkosten']:.2f} €")
    print(f"Einspeisevergütung: {result_kpis_rule['Einspeisevergütung']:.2f} €")
    wavg_price_rule = (result_df_rule["spotprice_EUR_per_MWh"] * result_df_rule["ee_export_MWh"]).sum() / max(result_kpis_rule["Einspeisung"], 1e-9)
    print(f"Export-gewichteter Preis (rule): {wavg_price_rule:.2f} €/MWh")
    print(f"Netzbezug: {result_kpis_rule['Netzbezug']:.2f} MWh")
    print(f"Einspeisung: {result_kpis_rule['Einspeisung']:.2f} MWh")
    print(f"Batterieladung: {result_kpis_rule['Batterieladung']:.2f} MWh")
    print(f"Batterieentladung: {result_kpis_rule['Batterieentladung']:.2f} MWh")
    print("----------------------")
    print("Optimiert (ohne Prognose):")
    print(f"Netto Stromkosten: {result_kpis_opt['Netto Stromkosten']:.2f} €")
    print(f"Netzstromkosten: {result_kpis_opt['Netzstromkosten']:.2f} €")
    print(f"Einspeisevergütung: {result_kpis_opt['Einspeisevergütung']:.2f} €")
    wavg_price_opt = (result_df_opt["spotprice_EUR_per_MWh"] * result_df_opt["ee_export_MWh"]).sum() / max(result_kpis_opt["Einspeisung"], 1e-9)
    print(f"Export-gewichteter Preis (opt):  {wavg_price_opt:.2f} €/MWh")
    print(f"Netzbezug: {result_kpis_opt['Netzbezug']:.2f} MWh")
    print(f"Einspeisung: {result_kpis_opt['Einspeisung']:.2f} MWh")
    print(f"Batterieladung: {result_kpis_opt['Batterieladung']:.2f} MWh")
    print(f"Batterieentladung: {result_kpis_opt['Batterieentladung']:.2f} MWh")
    print("----------------------")
    print("Optimiert (mit Prognose):")
    print(f"Netto Stromkosten: {result_kpis_opt_fore['Netto Stromkosten']:.2f} €")
    print(f"Netzstromkosten: {result_kpis_opt_fore['Netzstromkosten']:.2f} €")
    print(f"Einspeisevergütung: {result_kpis_opt_fore['Einspeisevergütung']:.2f} €")
    wavg_price_rl = (result_df_opt_fore["spotprice_EUR_per_MWh"] * result_df_opt_fore["ee_export_MWh"]).sum() / max(result_kpis_opt_fore["Einspeisung"], 1e-9)
    print(f"Export-gewichteter Preis (RL):  {wavg_price_rl:.2f} €/MWh")
    print(f"Netzbezug: {result_kpis_opt_fore['Netzbezug']:.2f} MWh")
    print(f"Einspeisung: {result_kpis_opt_fore['Einspeisung']:.2f} MWh")
    print(f"Batterieladung: {result_kpis_opt_fore['Batterieladung']:.2f} MWh")
    print(f"Batterieentladung: {result_kpis_opt_fore['Batterieentladung']:.2f} MWh")
    print("----------------------")



    # Plots
    import matplotlib.pyplot as plt
    import pandas as pd
    # Zeitspalten
    result_df_opt["datetime"] = pd.to_datetime(result_df_opt["datetime"])
    result_df_rule["datetime"] = pd.to_datetime(result_df_rule["datetime"])
    result_df_opt_fore["datetime"] = pd.to_datetime(result_df_opt_fore["datetime"])
    # Kopien + Hilfsspalten
    opt  = result_df_opt.copy()
    rule = result_df_rule.copy()
    rl   = result_df_opt_fore.copy()
    for df in (opt, rule, rl):
        df["grid_buy"]       = df["grid_to_load_MWh"] + df["grid_to_batt_MWh"]
        df["battery_state"]  = df["SOC_MWh"]
        df["grid_feed_in"]   = df["ee_export_MWh"]
        df["hourly_cost_EUR"] = (
            (df["grid_to_load_MWh"] + df["grid_to_batt_MWh"]) * df["spotprice_EUR_per_MWh"]
            - export_factor * df["ee_export_MWh"] * df["spotprice_EUR_per_MWh"]
        )
    # Zeitfenster
    start_time = opt["datetime"].min() + pd.Timedelta(days=1)
    end_time   = start_time + pd.Timedelta(days=3)
    opt  = opt[(opt["datetime"] >= start_time) & (opt["datetime"] < end_time)]
    rule = rule[(rule["datetime"] >= start_time) & (rule["datetime"] < end_time)]
    rl   = rl[(rl["datetime"] >= start_time) & (rl["datetime"] < end_time)]
    # Styles
    label_fontsize = 20
    tick_fontsize  = 20
    legend_fontsize= 20
    title_fontsize = 22
    c_opt  = "tab:blue"
    c_rule = "tab:orange"
    c_rl   = "tab:green"
    c_gen  = "tab:green"
    c_dem  = "tab:red"
    c_prc  = "tab:purple"
    # Plot (6 Panels inkl. Kosten)
    fig, axs = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Energieflüsse – optimiert vs. regelbasiert vs. RL", fontsize=title_fontsize+2)
    # 1) Netzbezug
    axs[0].set_title("Netzbezug", fontsize=title_fontsize)
    axs[0].plot(opt["datetime"],  opt["grid_buy"],  label="optimiert", color=c_opt, linewidth=2.5)
    axs[0].plot(rule["datetime"], rule["grid_buy"], label="regelbasiert", color=c_rule, linestyle="--", linewidth=2)
    axs[0].plot(rl["datetime"], rl["grid_buy"], label="RL", color=c_rl, linestyle=":", linewidth=2)
    axs[0].set_ylabel("Netzstrom [MWh]", fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    # 2) Einspeisung
    axs[1].set_title("Einspeisung", fontsize=title_fontsize)
    axs[1].plot(opt["datetime"],  opt["grid_feed_in"],  label="optimiert", color=c_opt, linewidth=2.5)
    axs[1].plot(rule["datetime"], rule["grid_feed_in"], label="regelbasiert", color=c_rule, linestyle="--", linewidth=2)
    axs[1].plot(rl["datetime"], rl["grid_feed_in"], label="RL", color=c_rl, linestyle=":", linewidth=2)
    axs[1].set_ylabel("Einspeisung [MWh]", fontsize=label_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    # 3) Batteriezustand
    axs[2].set_title("Batterieladestand (SOC)", fontsize=title_fontsize)
    axs[2].plot(opt["datetime"],  opt["battery_state"],  label="optimiert", color=c_opt, linewidth=2.5)
    axs[2].plot(rule["datetime"], rule["battery_state"], label="regelbasiert", color=c_rule, linestyle="--", linewidth=2)
    axs[2].plot(rl["datetime"], rl["battery_state"], label="RL", color=c_rl, linestyle=":", linewidth=2)
    axs[2].set_ylabel("SOC [MWh]", fontsize=label_fontsize)
    axs[2].legend(fontsize=legend_fontsize)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    # 4) Erzeugung und Verbrauch
    axs[3].set_title("Erzeugung und Verbrauch", fontsize=title_fontsize)
    axs[3].plot(opt["datetime"], opt["EE_total_MWh"], label="Erzeugung", color=c_gen, linewidth=2.5)
    axs[3].plot(opt["datetime"], opt["demand_MWh"],     label="Bedarf",    color=c_dem, linewidth=2)
    axs[3].set_ylabel("Energie [MWh]", fontsize=label_fontsize)
    axs[3].legend(fontsize=legend_fontsize)
    axs[3].tick_params(axis='both', labelsize=tick_fontsize)
    # 5) Strompreis
    axs[4].set_title("Strompreis", fontsize=title_fontsize)
    axs[4].plot(opt["datetime"], opt["spotprice_EUR_per_MWh"], label="Spot-Preis", color=c_prc, linewidth=2.5)
    axs[4].set_ylabel("Strompreis [€/MWh]", fontsize=label_fontsize)
    axs[4].legend(fontsize=legend_fontsize)
    axs[4].tick_params(axis='both', labelsize=tick_fontsize)
    # 6) Stündliche Stromkosten
    axs[5].set_title("Stündliche Stromkosten", fontsize=title_fontsize)
    axs[5].plot(opt["datetime"],  opt["hourly_cost_EUR"],  label="optimiert", color=c_opt, linewidth=2.5)
    axs[5].plot(rule["datetime"], rule["hourly_cost_EUR"], label="regelbasiert", color=c_rule, linestyle="--", linewidth=2)
    axs[5].plot(rl["datetime"], rl["hourly_cost_EUR"], label="RL", color=c_rl, linestyle=":", linewidth=2)
    axs[5].set_ylabel("Kosten [€/h]", fontsize=label_fontsize)
    axs[5].legend(fontsize=legend_fontsize)
    axs[5].tick_params(axis='both', labelsize=tick_fontsize)
    plt.xlabel("Date/time", fontsize=label_fontsize)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()