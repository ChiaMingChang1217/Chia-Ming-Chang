# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:38:31 2026

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fipy import Grid2D, CellVariable, DiffusionTerm


# =========================================================
# 0. 期刊風格設定
# =========================================================
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


# =========================================================
# 1. Plot helper: 關掉科學記號與 offset
# =========================================================
def disable_sci_notation(ax, axis="both"):
    ax.ticklabel_format(style='plain', axis=axis)
    if axis in ["x", "both"]:
        ax.xaxis.get_major_formatter().set_useOffset(False)
    if axis in ["y", "both"]:
        ax.yaxis.get_major_formatter().set_useOffset(False)


# =========================================================
# 2. 單一案例
# =========================================================
def run_case(nx, ny, V_app=0.10, L=5e-3, W=1e-3,
             k=1.46, sigma=1e5, S=200e-6,
             T_cold=300., T_hot=350., tol=1e-8,
             omega=0.5, max_iter=400):
    mesh = Grid2D(dx=W/nx, dy=L/ny, nx=nx, ny=ny)

    T = CellVariable(mesh=mesh, value=T_cold)
    V = CellVariable(mesh=mesh, value=0.0)
    Q = CellVariable(mesh=mesh, value=0.0)

    # Dirichlet BC
    T.constrain(T_hot,  mesh.facesTop)
    T.constrain(T_cold, mesh.facesBottom)
    V.constrain(V_app,  mesh.facesTop)
    V.constrain(0.0,    mesh.facesBottom)

    # 方程
    eqT = DiffusionTerm(coeff=k, var=T) == -Q
    eqV = DiffusionTerm(var=V) - DiffusionTerm(coeff=S, var=T) == 0

    converged = False
    last_res = np.nan

    # Picard iteration
    for it in range(max_iter):
        gradV = V.grad
        Q_new = sigma * (gradV[0]**2 + gradV[1]**2)
        Q.setValue(omega * Q_new + (1 - omega) * Q.value)

        resV = eqV.sweep(var=V, dt=0.5)
        resT = eqT.sweep(var=T, dt=1.0)
        last_res = max(abs(resV), abs(resT))

        if last_res < tol:
            converged = True
            break

    dT_max = float(T.value.max() - T_cold)
    Pin = float((Q.value * mesh.cellVolumes).sum())

    return {
        "nx": nx,
        "ny": ny,
        "cells": nx * ny,
        "dT_max": dT_max,
        "Pin": Pin,
        "n_iter": it + 1,
        "converged": converged,
        "final_residual": float(last_res),
        "dx": W / nx,
        "dy": L / ny
    }


# =========================================================
# 3. 收斂階數 p 與 GCI
# =========================================================
def calc_observed_order(f1, f2, f3, r):
    """
    f1: finest
    f2: medium
    f3: coarse
    r : refinement ratio
    """
    e21 = f2 - f1
    e32 = f3 - f2

    print(f"    e32 = {e32:.12e}, e21 = {e21:.12e}")

    # 若差太小，代表已幾乎完全收斂，observed p 難以穩定估計
    if abs(e21) < 1e-14 or abs(e32) < 1e-14:
        return np.nan

    ratio = e32 / e21

    # 若 <= 0，表示非單調收斂或有振盪，不適合直接用此公式
    if ratio <= 0:
        return np.nan

    return np.log(ratio) / np.log(r)


def calc_gci(f_fine, f_medium, r, p, Fs=1.25):
    """
    GCI for fine grid (%)
    """
    denom = r**p - 1.0
    if np.isnan(p) or abs(denom) < 1e-14:
        return np.nan
    return Fs * abs((f_medium - f_fine) / f_fine) / denom * 100.0


# =========================================================
# 4. Baseline
# =========================================================
print("Running Gałek-2018 baseline (60×300)...")
base = run_case(60, 300)
print(f"Baseline ΔT_max = {base['dT_max']:.6f} K, Pin = {base['Pin']:.6f} W")


# =========================================================
# 5. Grid-independence study
# 固定 refinement ratio = 1.5
# =========================================================
grid_list = [
    (20, 100),
    (30, 150),
    (45, 225),
    (60, 300),
    (90, 450),
    (120, 600)
]

records = []

print("\nRunning grid convergence study...")
for nx, ny in grid_list:
    result = run_case(nx, ny)
    records.append(result)
    print(f"{nx:3d}×{ny:<3d} | cells={result['cells']:6d} | "
          f"ΔT_max={result['dT_max']:10.6f} K | "
          f"Pin={result['Pin']:10.6f} W | "
          f"iter={result['n_iter']:3d} | conv={result['converged']}")

df = pd.DataFrame(records)

# =========================================================
# 6. 相對誤差（相對最細網格）
# =========================================================
fine_dT = df.iloc[-1]["dT_max"]
fine_Pin = df.iloc[-1]["Pin"]

df["rel_err_dT_%"] = abs(df["dT_max"] - fine_dT) / abs(fine_dT) * 100.0
df["rel_err_Pin_%"] = abs(df["Pin"] - fine_Pin) / abs(fine_Pin) * 100.0

# =========================================================
# 7. 用最後三層網格計算 observed order 與 GCI
# finest = 120x600, medium = 90x450, coarse = 60x300
# refinement ratio r = 1.5
# =========================================================
r = 1.5

f1_dT = df.iloc[-1]["dT_max"]   # finest = 120x600
f2_dT = df.iloc[-2]["dT_max"]   # medium = 90x450
f3_dT = df.iloc[-3]["dT_max"]   # coarse = 60x300

f1_Pin = df.iloc[-1]["Pin"]
f2_Pin = df.iloc[-2]["Pin"]
f3_Pin = df.iloc[-3]["Pin"]

print("\n--- Debug: last three grids for ΔT_max ---")
print(f"f3_dT (coarse) = {f3_dT:.12f}")
print(f"f2_dT (medium) = {f2_dT:.12f}")
print(f"f1_dT (fine)   = {f1_dT:.12f}")

print("\n--- Debug: last three grids for Pin ---")
print(f"f3_Pin (coarse) = {f3_Pin:.12f}")
print(f"f2_Pin (medium) = {f2_Pin:.12f}")
print(f"f1_Pin (fine)   = {f1_Pin:.12f}")

p_dT = calc_observed_order(f1_dT, f2_dT, f3_dT, r)
p_Pin = calc_observed_order(f1_Pin, f2_Pin, f3_Pin, r)

# 若 observed p 算不穩，fallback 用理論二階
if np.isnan(p_dT):
    p_dT = 2.0
    p_dT_source = "fallback_p=2"
else:
    p_dT_source = "observed"

if np.isnan(p_Pin):
    p_Pin = 2.0
    p_Pin_source = "fallback_p=2"
else:
    p_Pin_source = "observed"

gci_dT = calc_gci(f1_dT, f2_dT, r, p_dT)
gci_Pin = calc_gci(f1_Pin, f2_Pin, r, p_Pin)

# summary table
summary = pd.DataFrame({
    "metric": ["dT_max", "Pin"],
    "fine_value": [f1_dT, f1_Pin],
    "observed_order_p": [p_dT, p_Pin],
    "p_source": [p_dT_source, p_Pin_source],
    "GCI_fine_%": [gci_dT, gci_Pin]
})

# =========================================================
# 8. 輸出 CSV
# =========================================================
df.to_csv("grid_convergence.csv", index=False)
summary.to_csv("grid_gci_summary.csv", index=False)

# =========================================================
# 9. 收斂圖：ΔTmax
# =========================================================
h = 1.0 / np.sqrt(df["cells"])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(h, df["dT_max"],
        marker='o',
        markersize=6,
        linewidth=2,
        label=r"$\Delta T_{max}$")
ax.invert_xaxis()
ax.set_xlabel(r"$1/\sqrt{N_{cells}}$", fontsize=12)
ax.set_ylabel(r"$\Delta T_{max}$ (K)", fontsize=12)
ax.set_title(r"Grid Convergence of $\Delta T_{max}$", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False)
disable_sci_notation(ax, axis="both")
plt.tight_layout()
plt.savefig("grid_convergence_dT.png", dpi=300)
plt.show()

# =========================================================
# 10. 收斂圖：Pin
# =========================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(h, df["Pin"],
        marker='s',
        markersize=6,
        linewidth=2,
        label=r"$P_{in}$")
ax.invert_xaxis()
ax.set_xlabel(r"$1/\sqrt{N_{cells}}$", fontsize=12)
ax.set_ylabel(r"$P_{in}$ (W)", fontsize=12)
ax.set_title(r"Grid Convergence of $P_{in}$", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False)
disable_sci_notation(ax, axis="both")
plt.tight_layout()
plt.savefig("grid_convergence_Pin.png", dpi=300)
plt.show()

# =========================================================
# 11. 終端輸出
# =========================================================
print("\n================ Grid Convergence Summary ================")
print(df[["nx", "ny", "cells", "dT_max", "Pin", "rel_err_dT_%", "rel_err_Pin_%"]])

print("\n================ GCI Summary ================")
print(summary)

print("\n✓ Results saved:")
print("  - grid_convergence.csv")
print("  - grid_gci_summary.csv")
print("  - grid_convergence_dT.png")
print("  - grid_convergence_Pin.png")