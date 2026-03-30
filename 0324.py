# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:07:45 2026

@author: ASUS
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from fipy import Grid2D, CellVariable, DiffusionTerm


# =========================================================
# 1. Single electrothermal simulation
# =========================================================
def solve_TE(
    V_app,
    L,
    k_fac,
    k0=1.46,              # base thermal conductivity (W/m/K)
    sigma0=1.0e5,         # reference electrical conductivity at T0 (S/m)
    S0=2.0e-4,            # reference Seebeck coefficient at T0 (V/K)
    beta=8.0e-4,          # temperature coefficient for sigma(T)
    aS=-4.0e-8,           # slope of S(T), unit: V/K^2
    T0=300.0,             # reference temperature (K)
    sigma_min=1.0e4,      # lower bound for conductivity
    W=1.0e-3,             # width (m)
    nx=60,
    ny=300,
    T_cold=300.0,
    T_hot=350.0,
    omega=0.5,
    tol=1.0e-8,
    max_iter=500
):
    """
    Solve coupled electrothermal problem with:
        sigma(T) = sigma0 * [1 - beta * (T - T0)]
        S(T)     = S0 + aS * (T - T0)

    Returns
    -------
    dict
        dT_max, Pin, convergence info, average sigma/S values
    """

    # effective thermal conductivity
    k = k0 * k_fac

    # mesh
    mesh = Grid2D(dx=W / nx, dy=L / ny, nx=nx, ny=ny)

    # variables
    T = CellVariable(name="Temperature", mesh=mesh, value=T_cold)
    V = CellVariable(name="Voltage", mesh=mesh, value=0.0)
    Q = CellVariable(name="JouleHeat", mesh=mesh, value=0.0)

    sigma_var = CellVariable(name="Sigma(T)", mesh=mesh, value=sigma0)
    S_var = CellVariable(name="S(T)", mesh=mesh, value=S0)

    # boundary conditions
    T.constrain(T_hot, mesh.facesTop)
    T.constrain(T_cold, mesh.facesBottom)

    V.constrain(V_app, mesh.facesTop)
    V.constrain(0.0, mesh.facesBottom)

    # governing equations
    # Heat equation
    eqT = DiffusionTerm(coeff=k, var=T) == -Q

    # Voltage equation with temperature-dependent Seebeck coefficient
    eqV = DiffusionTerm(var=V) - DiffusionTerm(coeff=S_var, var=T) == 0

    converged = False
    last_res = np.nan

    for it in range(max_iter):
        # ---------------------------------------------
        # Update sigma(T)
        # ---------------------------------------------
        sigma_new = sigma0 * (1.0 - beta * (T.value - T0))
        sigma_new = np.maximum(sigma_new, sigma_min)
        sigma_var.setValue(sigma_new)

        # ---------------------------------------------
        # Update S(T)
        # ---------------------------------------------
        S_new = S0 + aS * (T.value - T0)
        S_var.setValue(S_new)

        # ---------------------------------------------
        # Joule heating
        # ---------------------------------------------
        gv = V.grad
        q_new = sigma_var.value * (gv[0].value**2 + gv[1].value**2)

        # under-relaxation
        Q.setValue(omega * q_new + (1.0 - omega) * Q.value)

        # sweep
        resV = eqV.sweep(var=V, dt=0.5)
        resT = eqT.sweep(var=T, dt=1.0)
        last_res = max(abs(resV), abs(resT))

        if last_res < tol:
            converged = True
            break

    dT_max = float(np.max(T.value) - T_cold)
    Pin = float(np.sum(Q.value * mesh.cellVolumes))

    return {
        "dT_max": dT_max,
        "Pin": Pin,
        "n_iter": int(it + 1),
        "converged": bool(converged),
        "final_residual": float(last_res),
        "sigma_avg": float(np.mean(sigma_var.value)),
        "sigma_min_case": float(np.min(sigma_var.value)),
        "sigma_max_case": float(np.max(sigma_var.value)),
        "S_avg": float(np.mean(S_var.value)),
        "S_min_case": float(np.min(S_var.value)),
        "S_max_case": float(np.max(S_var.value)),
    }


# =========================================================
# 2. DOE setup: full factorial 7200 cases
# =========================================================
bounds = {
    "V_app": (0.02, 0.10),   # 0.02 ~ 0.10 V
    "L": (1.0e-3, 1.0e-2),   # 1 ~ 10 mm
}

k_layers = [10]

N_V = 30
N_L = 30


def build_samples():
    """
    Build full-factorial samples:
        30 x 30 x 8 = 7200 cases
    """
    samples = []

    V_list = np.linspace(bounds["V_app"][0], bounds["V_app"][1], N_V)
    L_list = np.linspace(bounds["L"][0], bounds["L"][1], N_L)

    for kf in k_layers:
        for V_app in V_list:
            for L in L_list:
                samples.append({
                    "V_app": float(V_app),
                    "L": float(L),
                    "k_fac": float(kf)
                })

    return samples


# =========================================================
# 3. Utility: save dataframe
# =========================================================
def save_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")


# =========================================================
# 4. Main
# =========================================================
if __name__ == "__main__":

    samples = build_samples()
    total_cases = len(samples)

    print("=" * 70)
    print(f"🔄 Running {total_cases} cases in single-core mode ...")
    print("📁 Output: CSV only (no plotting)")
    print("📌 Checkpoint every 200 cases")
    print("=" * 70)

    results = []

    for i, s in enumerate(tqdm(samples, total=total_cases, ncols=90), start=1):
        try:
            out = solve_TE(
                V_app=s["V_app"],
                L=s["L"],
                k_fac=s["k_fac"],
                k0=1.46,
                sigma0=1.0e5,
                S0=2.0e-4,
                beta=8.0e-4,
                aS=-4.0e-8,
                T0=300.0,
                sigma_min=1.0e4,
                W=1.0e-3,
                nx=60,
                ny=300,
                T_cold=300.0,
                T_hot=350.0,
                omega=0.5,
                tol=1.0e-8,
                max_iter=500
            )

            row = {
                **s,
                **out,
                "error": ""
            }

        except Exception as e:
            row = {
                "V_app": s["V_app"],
                "L": s["L"],
                "k_fac": s["k_fac"],
                "dT_max": np.nan,
                "Pin": np.nan,
                "n_iter": np.nan,
                "converged": False,
                "final_residual": np.nan,
                "sigma_avg": np.nan,
                "sigma_min_case": np.nan,
                "sigma_max_case": np.nan,
                "S_avg": np.nan,
                "S_min_case": np.nan,
                "S_max_case": np.nan,
                "error": str(e),
            }

        results.append(row)

        # checkpoint every 200 cases
        if i % 200 == 0:
            df_temp = pd.DataFrame(results)
            save_csv(df_temp, "DOE_results_TdepSigmaS_checkpoint.csv")
            print(f"💾 Checkpoint saved at case {i}/{total_cases}")

    # -----------------------------------------------------
    # Save all results
    # -----------------------------------------------------
    df_all = pd.DataFrame(results)
    save_csv(df_all, "DOE_results_TdepSigmaS_all_7200.csv")

    # -----------------------------------------------------
    # Filter valid results
    # -----------------------------------------------------
    df_valid = df_all[
        df_all["dT_max"].notna() &
        df_all["Pin"].notna() &
        (df_all["dT_max"] > 0) &
        (df_all["dT_max"] <= 300) &
        (df_all["Pin"] > 0)
    ].copy()

    df_valid = df_valid.sort_values(by=["k_fac", "V_app", "L"]).reset_index(drop=True)
    save_csv(df_valid, "DOE_results_TdepSigmaS_valid_7200.csv")

    print("=" * 70)
    print(f"✅ Valid cases: {len(df_valid)} / {total_cases}")
    print("🎯 Simulation completed.")
    print("=" * 70)