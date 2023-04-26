import subprocess
import numpy as np

"0.5 1.0 1.0 0 0 0 "
specs = lambda S, h, delta: f"{S} {delta} {h} 0 0 0"

S = 1.5

# delta = 1.0
# h_vals = np.arange(0, S*4.0 + S*0.25, S*0.25)

h = 0.0
delta_vals = np.arange(-2.0, 2.0+0.25, 0.25)

# for i, h in enumerate(h_vals):
for i, delta in enumerate(delta_vals):
    subprocess.call(f"python3 exact_xxz_chain.py {specs(S, np.round(h, 6), np.round(delta, 6))}", shell=True)
