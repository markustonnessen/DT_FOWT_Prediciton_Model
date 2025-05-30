import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- load a trained MLSTM model using DOLPHINN on new FC2 data
"""

# Configure
TEST = "2a"
DOLPHINN_PATH = os.path.join("saved_models", "50Hz test 2", "wave_model")
DATA_PATH = os.path.join("data", "Combined_CS1-5_50Hz.csv")
PRESENT_TIME = 12000

if not os.path.exists(os.path.join("figures", f"{TEST}")):
    os.makedirs(os.path.join("figures", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)

# predict
data = pd.read_csv(DATA_PATH)
t1 = PRESENT_TIME
t2 = dol.time_horizon
t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
t2_idx = np.where(np.min(np.abs(data['Time']-(t2+t1))) == np.abs(data['Time']-(t2+t1)))[0][0]
state = data[dol.dof].mul(dol.conversion, axis=1).iloc[0:t1_idx]
time = data['Time'].iloc[0:t2_idx]
wave = data['wave'].iloc[0:t2_idx]
t_pred, y_hat = dol.predict(time, state, wave, history=500)

plt.figure()
plt.plot(time.iloc[0:t1_idx], state["PtfmTDZ"][0:t1_idx], color='black', label='Actual')
plt.plot(t_pred, y_hat["PtfmTDZ"], color='red', linestyle='-', label='Predicted')
plt.xlim((t1-250, t1+50))
plt.legend()
plt.savefig(os.path.join("figures", f"{TEST}", "test.pdf"), format="pdf")