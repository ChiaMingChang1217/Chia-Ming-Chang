# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:17:00 2025

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fipy import Grid2D, CellVariable, DiffusionTerm, ImplicitSourceTerm
from tqdm import tqdm
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count, freeze_support

# ---------- FiPy solver ----------
def solve_TE(V_app, L, k_fac, h_side, T_env=300., W=1e-3, nx=60,
             k0=1.46, sigma=1e5, S=200e-6,
             T_cold=300., T_hot=350., omega=0.5, tol=1e-8):

    ny = int(round(L * nx / W))  
    k = k0 * k_fac
    mesh = Grid2D(dx=W/nx, dy=L/ny, nx=nx, ny=ny)
    T = CellVariable(mesh=mesh, value=T_cold)
    V = CellVariable(mesh=mesh, value=0.)
    Q = CellVariable(mesh=mesh, value=0.)
    T.constrain(T_hot,  mesh.facesTop)
    T.constrain(T_cold, mesh.facesBottom)
    V.constrain(V_app,  mesh.facesTop)
    V.constrain(0.0,    mesh.facesBottom)
    side_cells = (mesh.cellCenters[0] < mesh.dx * 1.5) | (mesh.cellCenters[0] > W - mesh.dx * 1.5)
    side_mask = CellVariable(mesh=mesh, value=0.0)
    side_mask.setValue(1.0, where=side_cells)
    eqT = DiffusionTerm(k) + ImplicitSourceTerm(coeff=h_side * side_mask) == -Q + h_side * T_env * side_mask
    eqV = DiffusionTerm(var=V) - DiffusionTerm(coeff=S, var=T) == 0
    for _ in range(300):
        gv = V.grad
        Q.setValue(omega * sigma*(gv[0]**2 + gv[1]**2) + (1-omega)*Q.value)
        if max(eqV.sweep(V, dt=.5), eqT.sweep(T, dt=1.0)) < tol:
            break

    dT = float(T.value.max() - T_cold)
    Pin = float((Q.value * mesh.cellVolumes).sum())
    return {'V_app': V_app, 'L': L, 'k_fac': k_fac, 'h_side': h_side, 'dT_max': dT, 'Pin': Pin}

def solve_TE_wrapper(params):
    return solve_TE(**params)

if __name__ == "__main__":
    freeze_support()

    
    V_app_vals = np.linspace(0.02, 0.10, 30)
    L_vals     = np.linspace(1e-3, 1e-2, 30)
    k_fac_vals = [0.8, 1.0, 1.2, 1.5, 2, 3, 5, 10]
    h_vals     = [0]

    samples = [{'V_app': v, 'L': l, 'k_fac': k, 'h_side': h}
               for h in h_vals for k in k_fac_vals for l in L_vals for v in V_app_vals]

    print(f"🔁 Running grid with {len(samples)} cases using {cpu_count()} cores …")
    with Pool(processes=cpu_count()) as pool:
        records = list(tqdm(pool.imap_unordered(solve_TE_wrapper, samples),
                            total=len(samples), ncols=70, desc="Sim"))

    df = pd.DataFrame(records)
    df.to_csv("Grid_results.csv", index=False)
    print("✅ Grid_results.csv saved.")

    k_layers = [0.8, 1, 1.2]
    grid_V = np.linspace(0.02, 0.10, 60)
    grid_L = np.linspace(1e-3, 1e-2, 60)
    Vg, Lg = np.meshgrid(grid_V, grid_L)

    for kf in k_layers:
        for h in h_vals:
            sel = df[(np.isclose(df['k_fac'], kf, atol=1e-4)) &
                     (np.isclose(df['h_side'], h, atol=1e-4))]
            if sel.empty:
                continue

            points = np.column_stack((sel['V_app'], sel['L']))
            Z_dT   = griddata(points, sel['dT_max'], (Vg, Lg), method='cubic')
            Z_Pin  = griddata(points, sel['Pin'],   (Vg, Lg), method='cubic')

            
            plt.figure(figsize=(5,4))
            cs = plt.contourf(Vg, Lg*1e3, Z_dT, 20, cmap='magma')
            plt.colorbar(cs, label='ΔT_max (K)')
            plt.xlabel('V_app (V)'); plt.ylabel('L (mm)')
            plt.title(f'ΔT_max map (k_fac={kf:.2f}, h={h})')
            plt.tight_layout(); plt.savefig(f'Fig4a_dT_k{kf:.2f}_h{h}.png', dpi=300)
            plt.close()

            plt.figure(figsize=(5,4))
            cs2 = plt.contourf(Vg, Lg*1e3, Z_Pin, 20, cmap='viridis')
            plt.colorbar(cs2, label='Pin (W)')
            for level, color, label in zip([100, 80, 60, 40, 20],
                                           ['pink', 'red', 'blue', 'yellow', 'green'],
                                           ['ΔT=100K', 'ΔT=80K', 'ΔT=60K', 'ΔT=40K', 'ΔT=20K']):
                c = plt.contour(Vg, Lg*1e3, Z_dT, levels=[level], colors=color, linewidths=2, linestyles='--')
                plt.clabel(c, fmt=label, colors=color)
            plt.xlabel('V_app (V)'); plt.ylabel('L (mm)')
            plt.title(f'Pin map (k_fac={kf:.2f}, h={h})')
            plt.tight_layout(); plt.savefig(f'Fig4b_Pin_k{kf:.2f}_h{h}.png', dpi=300)
            plt.close()

            print(f"📊：Fig4a_dT_k{kf:.2f}_h{h}.png & Fig4b_Pin_k{kf:.2f}_h{h}.png")

    print("🎯 All plots done.")
