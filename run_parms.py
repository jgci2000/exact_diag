import subprocess
import numpy as np

"0.5 1.0 1.0 0 0 0 "
specs = lambda S, h, delta: f"{S} {delta} {h} 0 0 0"

S = 0.5
delta = 1.0

h_vals_1 = np.arange(0.0, S*1.0, S*0.025)
h_vals_2 = np.arange(S*1.0, S*3.0, S*0.25)
h_vals_3 = np.arange(S*3.0, S*4.0, S*0.025)

h_vals = np.concatenate([h_vals_1, h_vals_2, h_vals_3, [S*4.0]])

for i, h in enumerate(h_vals):
    subprocess.call(f"python3 exact_xxz_chain.py {specs(S, np.round(h, 6), delta)}", shell=True)
