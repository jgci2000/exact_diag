import numpy as np
from itertools import product

N = 2

SPIN = [1/2, -1/2]
ALL_STATES = list(product(SPIN, repeat=N))
N_STATES = len(ALL_STATES)

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

H = np.zeros((N_STATES, N_STATES))
for i, bra in enumerate(ALL_STATES):
    for j, ket in enumerate(ALL_STATES):
        H[i, j] = H_term(bra, ket)

print(f"{ALL_STATES=}")
print(f"{H=}")


