# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:49:28 2026

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid2D, CellVariable, DiffusionTerm

# =========================
# Gałek baseline + T-dependent properties
# =========================

# --- 幾何與網格 ---
W = 1e-3       # width: 1 mm
L = 5e-3       # height: 5 mm
nx, ny = 60, 300
dx, dy = W / nx, L / ny

# --- 基本材料參數 (at T0) ---
k = 1.46                # W/m·K
sigma0 = 1e5            # S/m at T0
S0 = 200e-6             # V/K at T0
T0 = 300.0              # reference temperature (K)

# --- 溫度相關係數 ---
# 依照你論文目前描述：
# sigma 在 300–350 K 約下降 4%  => beta = 0.04 / 50 = 8e-4 1/K
# S 在 300–350 K 約下降 1%      => dS = -0.01*S0 over 50 K
beta = 8.0e-4           # 1/K
aS = -4.0e-8            # V/K^2

# --- 邊界條件 ---
T_hot = 350.0
T_cold = 300.0
V_app = 0.06

# --- 數值參數 ---
omega = 0.5
tol = 1e-8
max_iter = 500

# =========================
# 建立網格與變數
# =========================
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

T = CellVariable(mesh=mesh, name="Temperature", value=T_cold)
V = CellVariable(mesh=mesh, name="Potential", value=0.0)
Q = CellVariable(mesh=mesh, name="JouleHeat", value=0.0)

sigma_var = CellVariable(mesh=mesh, name="sigma(T)", value=sigma0)
S_var = CellVariable(mesh=mesh, name="S(T)", value=S0)

# =========================
# 邊界條件
# =========================
T.constrain(T_hot, mesh.facesTop)
T.constrain(T_cold, mesh.facesBottom)

V.constrain(V_app, mesh.facesTop)
V.constrain(0.0, mesh.facesBottom)

# =========================
# 迭代求解
# =========================
residuals = []

for i in range(max_iter):
    # --- 更新溫度相關材料性質 ---
    sigma_new = sigma0 * (1.0 - beta * (T.value - T0))
    # 避免數值上變成負值或過小
    sigma_new = np.clip(sigma_new, 1e-12, None)
    sigma_var.setValue(sigma_new)

    S_new = S0 + aS * (T.value - T0)
    S_var.setValue(S_new)

    # --- 電場更新後的 Joule heating ---
    gradV = V.grad.value
    joule_raw = sigma_var.value * (gradV[0]**2 + gradV[1]**2)

    # under-relaxation
    Q.setValue(omega * joule_raw + (1.0 - omega) * Q.value)

    # --- 電位方程 ---
    # ∇·(σ ∇V) - ∇·(σ S ∇T) = 0
    eqV = (
        DiffusionTerm(coeff=sigma_var, var=V)
        - DiffusionTerm(coeff=sigma_var * S_var, var=T)
        == 0
    )

    # --- 溫度方程 ---
    # ∇·(k ∇T) + Q = 0  -> DiffusionTerm(k) == -Q
    eqT = DiffusionTerm(coeff=k, var=T) == -Q

    # sweep
    resV = eqV.sweep(var=V)
    resT = eqT.sweep(var=T)

    res = max(float(resV), float(resT))
    residuals.append(res)

    if res < tol:
        print(f"Converged at iteration {i+1}, residual = {res:.3e}")
        break
else:
    print(f"Did not fully converge within {max_iter} iterations. Final residual = {residuals[-1]:.3e}")

# =========================
# 能量平衡
# =========================
joule_total = float((Q * mesh.cellVolumes).value.sum())

mask_top = mesh.facesTop.value
mask_bot = mesh.facesBottom.value

ny_top = np.abs(mesh.faceNormals[1][mask_top])
ny_bot = np.abs(mesh.faceNormals[1][mask_bot])

area_top = mesh._faceAreas[mask_top] * ny_top
area_bottom = mesh._faceAreas[mask_bot] * ny_bot

gradTy_top = T.faceGrad.dot([0, 1]).value[mask_top]
gradTy_bot = T.faceGrad.dot([0, 1]).value[mask_bot]

q_top = -k * np.sum(gradTy_top * area_top)
q_bottom = k * np.sum(gradTy_bot * area_bottom)

q_boundary = q_top + q_bottom
err_pct = abs(joule_total - q_boundary) / max(abs(joule_total), 1e-20) * 100.0

print(f"Joule heat total   : {joule_total:.6e} W")
print(f"Boundary heat flux : {q_boundary:.6e} W")
print(f"Energy balance err : {err_pct:.4f} %")
print(f"sigma(T_hot≈350K)  : {sigma0 * (1 - beta*(350-T0)):.3f} S/m")
print(f"S(T_hot≈350K)      : {S0 + aS*(350-T0):.6e} V/K")
# =========================
# Validation: Tmax error vs reference
# =========================

Tmax_sim = float(T.value.max())

# Gałek 2018 reference (你之後可以改)
Tmax_ref = 354.0   # ← 這裡請填正確 reference 值

abs_err = abs(Tmax_sim - Tmax_ref)
rel_err = abs_err / Tmax_ref * 100.0

print("\n--- Validation (Tmax) ---")
print(f"Tmax (simulation) : {Tmax_sim:.3f} K")
print(f"Tmax (reference)  : {Tmax_ref:.3f} K")
print(f"Absolute error    : {abs_err:.3f} K")
print(f"Relative error    : {rel_err:.3f} %")

# =========================
# 資料整理
# =========================
T_map = np.reshape(T.value, (ny, nx))
V_map = np.reshape(V.value, (ny, nx))
sigma_map = np.reshape(sigma_var.value, (ny, nx))
S_map = np.reshape(S_var.value, (ny, nx))

x = np.linspace(0, W * 1e3, nx)
y = np.linspace(0, L * 1e3, ny)

# =========================
# 溫度場
# =========================
plt.figure(figsize=(6, 4))
plt.imshow(
    T_map,
    extent=[0, W * 1e3, 0, L * 1e3],
    origin="lower",
    cmap="inferno",
    interpolation="bicubic",
    aspect="auto"
)
plt.colorbar(label="Temperature (K)")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Temperature Field")
plt.tight_layout()
plt.savefig("baseline_T_Tdep.png", dpi=300)
plt.show()

# =========================
# 電位場
# =========================
plt.figure(figsize=(6, 4))
plt.imshow(
    V_map,
    extent=[0, W * 1e3, 0, L * 1e3],
    origin="lower",
    cmap="viridis",
    interpolation="bicubic",
    aspect="auto"
)
plt.colorbar(label="Electric Potential (V)")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Electric Potential Field")
plt.tight_layout()
plt.savefig("baseline_V_Tdep.png", dpi=300)
plt.show()

# =========================
# sigma(T) 分布
# =========================
plt.figure(figsize=(6, 4))
plt.imshow(
    sigma_map,
    extent=[0, W * 1e3, 0, L * 1e3],
    origin="lower",
    cmap="plasma",
    interpolation="bicubic",
    aspect="auto"
)
plt.colorbar(label="Electrical Conductivity (S/m)")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Temperature-Dependent Conductivity Field")
plt.tight_layout()
plt.savefig("baseline_sigma_Tdep.png", dpi=300)
plt.show()

# =========================
# S(T) 分布
# =========================
plt.figure(figsize=(6, 4))
plt.imshow(
    S_map,
    extent=[0, W * 1e3, 0, L * 1e3],
    origin="lower",
    cmap="cividis",
    interpolation="bicubic",
    aspect="auto"
)
plt.colorbar(label="Seebeck Coefficient (V/K)")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Temperature-Dependent Seebeck Field")
plt.tight_layout()
plt.savefig("baseline_Seebeck_Tdep.png", dpi=300)
plt.show()

# =========================
# 殘差圖
# =========================
plt.figure(figsize=(6, 4))
plt.semilogy(residuals, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("Convergence History")
plt.tight_layout()
plt.savefig("baseline_residual_Tdep.png", dpi=300)
plt.show()