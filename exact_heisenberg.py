import numpy as np
import numpy.linalg as npla
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
HX = 5
HY = 4

SAVEFOLDER = "figures/"
OUTPUT_DIR = "SSE_results/"

Sz = lambda state : state[0] if state[0] == state[1] else 0.0
Sm = lambda state : 1.0 if state[1] > state[0] else 0.0
Sp = lambda state : 1.0 if state[1] < state[0] else 0.0

def H_term(bra, ket):
    term = 0.0
    
    for i in range(N):
        j = (i + 1) % N
        
        term += Sz((bra[i], ket[i])) * Sz((bra[j], ket[j]))
        term += 0.5 * (Sp((bra[i], ket[i])) * Sm((bra[j], ket[j])) + Sm((bra[i], ket[i])) * Sp((bra[j], ket[j])))
    
    return term

SPIN = [1/2, -1/2]
COLOR = ["r", "b", "g", "y", "k"]

for c, N in enumerate([2, 4, 6]):
    ALL_STATES = list(product(SPIN, repeat=N))
    N_STATES = len(ALL_STATES)

    H = np.zeros((N_STATES, N_STATES))
    for i, bra in enumerate(ALL_STATES):
        for j, ket in enumerate(ALL_STATES):
            H[i, j] = H_term(bra, ket)

    eig_vals, eig_kets = npla.eigh(H)
    diag_H = np.diag(eig_vals)
    
    print(f"all states: {ALL_STATES}")
    print(f"ground state energy: {eig_vals[0]}")
    print(f"ground state: {eig_kets[:, 0]}")

    # print(pd.DataFrame(H))
    # print(pd.DataFrame(eig_vals))
    
    # T = np.arange(0.01, 2.0, 0.01)
    # T_vals = len(T)
    # beta = 1.0 / T
    beta = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    T_vals = len(beta)
    T = 1.0 / beta

    E_vals = eig_vals

    Z = np.array([np.sum(np.exp(- beta[i] * E_vals)) for i in range(T_vals)])

    E = np.array([np.sum(E_vals * np.exp(- beta[i] * E_vals)) / Z[i] for i in range(T_vals)])
    E2 = np.array([np.sum(E_vals**2 * np.exp(- beta[i] * E_vals)) / Z[i] for i in range(T_vals)])
    C = np.array([beta[i]**2 * (E2[i] - E[i]**2) for i in range(T_vals)])

    E /= N
    C /= N

    T_vals_sim = 6
    beta_sim = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    T_sim = 1.0 / beta_sim
    
    E_sim = np.zeros(T_vals_sim)
    E_std = np.zeros(T_vals_sim)
    C_sim = np.zeros(T_vals_sim)
    C_std = np.zeros(T_vals_sim)
    m_sim = np.zeros(T_vals_sim)
    m_std = np.zeros(T_vals_sim)

    with open(OUTPUT_DIR + f"output_N{N}.csv", "r") as f:
        header = f.readline()
        for j in range(T_vals_sim):
            _, _, _, _, E_sim[j], E_std[j], C_sim[j], C_std[j], m_sim[j], m_std[j], _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = [float(x) for x in f.readline().strip().split(",")]

    print(f"{N=}")
    print("beta |  E Exact  |  SSE +/- std  |  in?")
    for i in range(T_vals):
        print(f"{beta[i]}  |  {E[i]}  |  {E_sim[i]} +/- {E_std[i]}  |  {True if (E[i] < E_sim[i] + E_std[i] and E[i] > E_sim[i] - E_std[i]) else False}")
    print("beta |  C Exact  |  SSE +/- std  |  in?")
    for i in range(T_vals):
        print(f"{beta[i]}  |  {C[i]}  |  {C_sim[i]} +/- {C_std[i]}  |  {True if (C[i] < C_sim[i] + C_std[i] and C[i] > C_sim[i] - C_std[i]) else False}")
    print()

    plt.figure(1, figsize=(HX, HY))
    plt.plot(T, E, "-" + COLOR[c],label=f"{N=}")
    plt.errorbar(T_sim, E_sim, E_std, fmt="--" + COLOR[c])
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\langle E \rangle$")
    plt.legend()

    plt.figure(2, figsize=(HX, HY))
    plt.plot(T, C, "-" + COLOR[c],label=f"{N=}")
    plt.errorbar(T_sim, C_sim, C_std, fmt="--" + COLOR[c])
    plt.xlabel(r"$T$")
    plt.ylabel(r"$C$")
    plt.legend()

plt.show()
