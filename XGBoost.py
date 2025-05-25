# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:46:11 2025

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from SALib.sample import saltelli
from SALib.analyze import sobol

df = pd.read_csv("Grid_results.csv")
X = df[["V_app", "L", "k_fac"]].values
y_dT = df["dT_max"].values
y_Pin = df["Pin"].values

X_train, X_test, y_dT_train, y_dT_test = train_test_split(X, y_dT, test_size=0.2, random_state=42)
_, _, y_Pin_train, y_Pin_test = train_test_split(X, y_Pin, test_size=0.2, random_state=42)

model_dt = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model_dt.fit(X_train, y_dT_train)

model_pin = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model_pin.fit(X_train, y_Pin_train)

print(f"ΔT_max R2:", round(r2_score(y_dT_test, model_dt.predict(X_test)), 4))
print(f"Pin     R2:", round(r2_score(y_Pin_test, model_pin.predict(X_test)), 4))

problem = {
    'num_vars': 3,
    'names': ['V_app', 'L', 'k_fac'],
    'bounds': [[0.02, 0.10], [1e-3, 1e-2], [0.8, 10.0]]
}

param_values = saltelli.sample(problem, 2048, calc_second_order=False)
y_dt_sobol = model_dt.predict(param_values)
y_pin_sobol = model_pin.predict(param_values)

Si_dt = sobol.analyze(problem, y_dt_sobol, calc_second_order=False)
Si_pin = sobol.analyze(problem, y_pin_sobol, calc_second_order=False)

def plot_sobol(Si, title, filename):
    names = problem['names']
    S1 = Si['S1']
    ST = Si['ST']
    S1_conf = Si['S1_conf']
    ST_conf = Si['ST_conf']
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.15, S1, yerr=S1_conf, width=0.3, label='S1', color='skyblue', capsize=4)
    ax.bar(x + 0.15, ST, yerr=ST_conf, width=0.3, label='ST', color='orange', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Sensitivity Index")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
plot_sobol(Si_dt, "Sobol Sensitivity for ΔT_max", "Sobol_dTmax_errorbar.png")
plot_sobol(Si_pin, "Sobol Sensitivity for Pin", "Sobol_Pin_errorbar.png")
