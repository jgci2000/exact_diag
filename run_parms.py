import subprocess
import numpy as np

"0.5 1.0 1.0 0 0 0 "
specs = lambda S, h, delta: f"{S} {delta} {h} 0 0 0"

S = 0.5
delta = 1.0
h_vals = np.arange(0.0, S*4.0 + S*0.025, S*0.025)

for i, h in enumerate(h_vals):
    subprocess.call(f"python3 exact_xxz_chain.py {specs(S, h, delta)}", shell=True)
