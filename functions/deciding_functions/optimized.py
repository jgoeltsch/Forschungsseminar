import pandas as pd
import cvxpy as cp

def optimize_energy_flow(df, battery_capacity, initial_battery, price_buy=100):
    
    print("⚙️ Verfügbare Solver:", cp.installed_solvers())

    n = len(df)
    demand = df["energy_demand"].values
    generation = df["total_energy_production"].values

    battery_charge = cp.Variable(n)
    battery_discharge = cp.Variable(n)
    grid_buy = cp.Variable(n)
    curtailment = cp.Variable(n)
    battery_state = cp.Variable(n + 1)

    is_charging = cp.Variable(n, boolean=True)
    is_discharging = cp.Variable(n, boolean=True)
    max_rate = 12  # Lade-/Entladerate (je nach Setup)


    constraints = [battery_state[0] == initial_battery]

    for t in range(n):
    # Energieflussbilanz
        constraints.append(
            generation[t] + grid_buy[t] + battery_discharge[t] ==
            demand[t] + battery_charge[t] + curtailment[t]
        )

        # Batterie-Zustandsdynamik
        constraints.append(
            battery_state[t + 1] == battery_state[t] + battery_charge[t] - battery_discharge[t]
        )

        # Nichtnegative Variablen & Batteriegrenzen
        constraints += [
            0 <= battery_charge[t],
            0 <= battery_discharge[t],
            0 <= grid_buy[t],
            0 <= curtailment[t],
            0 <= battery_state[t + 1],
            battery_state[t + 1] <= battery_capacity
        ]

        # Neue MIP-Constraints zur Verhinderung von gleichzeitiger Ladung & Entladung
        constraints += [
            battery_charge[t] <= max_rate * is_charging[t],
            battery_discharge[t] <= max_rate * is_discharging[t],
            is_charging[t] + is_discharging[t] <= 1
        ]


    total_cost = cp.sum(grid_buy * price_buy)
    problem = cp.Problem(cp.Minimize(total_cost), constraints)

    for solver in ["GUROBI", "CBC", "GLPK_MI", "ECOS_BB"]:
        if solver in cp.installed_solvers():
            print(f"👉 Versuche Solver: {solver}")
            try:
                problem.solve(solver=solver)
                print("✅ Solver erfolgreich:", solver)
                break
            except Exception as e:
                print(f"❌ Fehler bei Solver {solver}: {e}")
    else:
        raise RuntimeError("Kein geeigneter MIP-Solver installiert.")

    # Statusprüfung ergänzen
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimierung fehlgeschlagen. Status: {problem.status}")
    print("Optimierungsstatus:", problem.status)

    result_df_opt = df.copy()
    result_df_opt["grid_buy"] = grid_buy.value
    result_df_opt["battery_charge"] = battery_charge.value
    result_df_opt["battery_discharge"] = battery_discharge.value
    result_df_opt["battery_state"] = battery_state.value[:-1]
    result_df_opt["curtailment"] = curtailment.value
    result_df_opt["costs"] = result_df_opt["grid_buy"] * price_buy
    result_df_opt["saldo"] = result_df_opt["costs"]
    total_cost = result_df_opt["saldo"].sum()

    return result_df_opt, total_cost
