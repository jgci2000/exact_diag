import numpy as np
import numpy.linalg as npla
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import os, sys
import pandas as pd

plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
COLOR = ["r", "b", "g", "y", "k"]
HX = 5
HY = 4

SAVEFOLDER = "figures/"
SSE_DIR = "sse_results/"
EXACT_DIR = "exact_results/"

N = 8
J = 1.0
DELTA = 1.0
H = [0.0, 0.25, 0.5, 1.0, 2.0]
H_VALS = len(H)

T_vals_sim = 6
beta_sim = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
T_sim = 1.0 / beta_sim

T = np.arange(0.01, 2.0, 0.01)
T_vals = len(T)
beta = 1.0 / T

E_sim = np.zeros((H_VALS, T_vals_sim))
E_std = np.zeros((H_VALS, T_vals_sim))
C_sim = np.zeros((H_VALS, T_vals_sim))
C_std = np.zeros((H_VALS, T_vals_sim))
m_sim = np.zeros((H_VALS, T_vals_sim))
m_std = np.zeros((H_VALS, T_vals_sim))
m2_sim = np.zeros((H_VALS, T_vals_sim))
m2_std = np.zeros((H_VALS, T_vals_sim))
m_sus_sim = np.zeros((H_VALS, T_vals_sim))
m_sus_std = np.zeros((H_VALS, T_vals_sim))

for i in range(H_VALS):
    with open(SSE_DIR + f"output_N{N}_delta{DELTA}_h{H[i]}.csv", "r") as f:
        header = f.readline()
        for j in range(T_vals_sim):
            _, _, _, _, E_sim[i, j], E_std[i, j], C_sim[i, j], C_std[i, j], m_sim[i, j], m_std[i, j], m2_sim[i, j], m2_std[i, j], _, _, _, _, _, _, _, _, m_sus_sim[i, j], m_sus_std[i, j], _, _, _, _ = [float(x) for x in f.readline().strip().split(",")]

E = np.zeros((H_VALS, T_vals))
C = np.zeros((H_VALS, T_vals))
m = np.zeros((H_VALS, T_vals))
m2 = np.zeros((H_VALS, T_vals))
m_sus = np.zeros((H_VALS, T_vals))

for i in range(H_VALS):
    with open(EXACT_DIR + f"exact_N{N}_delta{DELTA}_h{H[i]}.csv", "r") as f:
        header = f.readline()
        for j in range(T_vals):
            _, E[i, j], C[i, j], m[i, j], m2[i, j], m_sus[i, j] = [float(x) for x in f.readline().strip().split(",")]

plt.figure(1, figsize=(HX, HY))
plt.title("SSE")
for h_indx in range(H_VALS):
    plt.errorbar(T_sim, E_sim[h_indx, :], E_std[h_indx, :], fmt=".-", label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle E \rangle$")
plt.legend()

plt.figure(2, figsize=(HX, HY))
plt.title("Exact")
for h_indx in range(H_VALS):
    plt.plot(T, E[h_indx, :], label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle E \rangle$")
plt.legend()

plt.figure(3, figsize=(HX, HY))
plt.title("SSE")
for h_indx in range(H_VALS):
    plt.errorbar(T_sim, m_sim[h_indx, :], m_std[h_indx, :], fmt=".-", label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle m \rangle$")
plt.legend()

plt.figure(4, figsize=(HX, HY))
plt.title("Exact")
for h_indx in range(H_VALS):
    plt.plot(T, m[h_indx, :], label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle m \rangle$")
plt.legend()

plt.figure(5, figsize=(HX, HY))
plt.title("SSE")
for h_indx in range(H_VALS):
    plt.errorbar(T_sim, m2_sim[h_indx, :], m2_std[h_indx, :], fmt=".-", label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle m^2 \rangle$")
plt.legend()

plt.figure(6, figsize=(HX, HY))
plt.title("Exact")
for h_indx in range(H_VALS):
    plt.plot(T, m2[h_indx, :], label=f"h={H[h_indx]}")
plt.xlabel(r"$T$")
plt.ylabel(r"$\langle m^2 \rangle$")
plt.legend()

plt.show()
