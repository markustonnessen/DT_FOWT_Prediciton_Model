import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===================================================
# =============== EDITABLE PARAMETERS ===============
# ===================================================

## --- SIMA COMMAND INPUT PARAMETERS (commandsModel.txt) ---
heading = 180
xref = 25.875
yref = -62.817
hs = 3
tp = 11
xhook = 25.875
yhook = -44.817
SimLen = 2000
WavDir = 30

## --- MLSTM & PREDICTION PARAMETERS ---
MLSTM_MODEL_NAME = "Option2_WD30_Floater_Case3"
Prediction_state = "PtfmTDX_Floater"
CaseNr = "Case3"
plot_figure = True

timestep = 0.25
time_horizon = 20.0
pred_error_x = 0.0
pred_error_y = 0.0
pred_freq = 5.0
save_csv = True
save_csv_time = 590
early_stop_enabled = True
early_stop_time = 600

# ===================================================
# === INTERNAL PATHS & FILES ========================
# ===================================================

pathToSima = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation\sima-4.8.0-windows\sima.exe"
pathToWorkspace = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation"
pathToCommandFile = os.path.join(pathToWorkspace, "commandsModel.txt")
args_path = os.path.join(pathToWorkspace, "sima_args.txt")

# ===================================================
# === UPDATE SIMA COMMAND FILE ======================
# ===================================================

with open(pathToCommandFile, "r") as file:
    lines = file.readlines()

lines[1] = (
    f"run task=Workflow1 workflow=Dynamic input="
    f"Heading={heading};Xref={xref};Yref={yref};Hs={hs};Tp={tp};"
    f"Xhook={xhook};Yhook={yhook};SimLen={SimLen};WavDir={WavDir}\n"
)

with open(pathToCommandFile, "w") as file:
    file.writelines(lines)

# ===================================================
# === PLOT FLOATER & POSITIONS ======================
# ===================================================

distance = 80.8 / 2
x_front = xref + distance * np.cos(np.radians(heading))
y_front = yref + distance * np.sin(np.radians(heading))
x_aft = xref - distance * np.cos(np.radians(heading))
y_aft = yref - distance * np.sin(np.radians(heading))

fig, ax = plt.subplots()
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect('equal')
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_xlabel("Y-axis")
ax.set_ylabel("X-axis")
ax.set_xticks(range(-100, 101, 10))
ax.set_yticks(range(-100, 101, 10))

pontoon_radius = 12.5 / 2
ax.add_patch(patches.Circle((0, 0), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2))
ax.add_patch(patches.Circle((-51.75, 0), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2))
ax.add_patch(patches.Circle((25.875, 44.817), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2))
ax.add_patch(patches.Circle((25.875, -44.817), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2))

ax.plot([0, -51.75], [0, 0], color='purple', linewidth=2)
ax.plot([0, 25.875], [0, 44.817], color='purple', linewidth=2)
ax.plot([0, 25.875], [0, -44.817], color='purple', linewidth=2)
ax.annotate('', xy=(x_front, y_front), xytext=(xref, yref), arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', linewidth=2),)
ax.scatter([xref, xhook], [yref, yhook], color='red', s=50)
ax.plot([xref, xhook], [yref, yhook], color='green', linewidth=2)
ax.plot([xref, x_front], [yref, y_front], color='blue', linewidth=2)


ax.plot([xref, x_aft], [yref, y_aft], color='blue', linewidth=2)

gangway_distance = np.sqrt((xhook - xref) ** 2 + (yhook - yref) ** 2)
print(gangway_distance)
plt.title("SIMA Setup Layout")
plt.show()

# ===================================================
# === SAVE ARGS TO sima_args.txt FOR MLSTM =========
# ===================================================

params = {
    "MLSTM_MODEL_NAME": MLSTM_MODEL_NAME,
    "timestep": timestep,
    "plot_figure": plot_figure,
    "time_horizon": time_horizon,
    "pred_error_x": pred_error_x,
    "pred_error_y": pred_error_y,
    "pred_freq": pred_freq,
    "save_csv": save_csv,
    "save_csv_time": save_csv_time,
    "Prediction_state": Prediction_state,
    "early_stop_enabled": early_stop_enabled,
    "early_stop_time": early_stop_time,
    "WavDir": WavDir,
    "CaseNr": CaseNr
}

with open(args_path, "w") as f:
    for key, val in params.items():
        f.write(f"{key}={repr(val)}\n")

# ===================================================
# ================= LAUNCH SIMA =====================
# ===================================================

subprocess.check_call([
    pathToSima,
    "-application", "no.marintek.sima.application.headless.application",
    "-consoleLog",
    "-noSplash",
    "-data", pathToWorkspace,
    "-commands", "file=" + pathToCommandFile
], shell=True)

print("SIMA has run without errors.")
