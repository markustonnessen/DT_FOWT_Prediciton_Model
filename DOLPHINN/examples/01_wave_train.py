import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- train MLSTM model using DOLPHINN on FC2 data
"""

# Configure
TEST = "Option2_WD30_SOV_Case3"  
CONFIG_FILE_PATH = os.path.join("DOLPHINN", "dol_input", "wave.yaml")
if not os.path.exists(os.path.join("DOLPHINN","training_results", f"{TEST}")):
    os.makedirs(os.path.join("DOLPHINN","training_results", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.train(config_path=CONFIG_FILE_PATH)
r_square, mae, y, y_hat = dol.test()

# post-processing
fig = plt.figure(figsize=(12, 24))
gs = gridspec.GridSpec(len(dol.dof), 1)

for i, (label, unit) in enumerate(zip(dol.dof, dol.unit)):

    ax = plt.subplot(gs[i])
    ax.plot(y[:, i], label='experiment', color='black')
    ax.plot(y_hat[:, i], label='DOLPHINN', color='red', linestyle='--')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(f"{label} {unit}")
    ax.set_xlim((1000, 1250))
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()
plt.savefig(os.path.join("DOLPHINN","training_results", f"{TEST}", f"wave_{TEST}.pdf"), format="pdf", dpi=300)

# save dolphinn
dol.save(os.path.join("DOLPHINN", "saved_models", f"{TEST}", "wave_model"))