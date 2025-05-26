import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# === CONFIGURATION ===
base_dir = os.path.join(os.path.dirname(__file__), "prediction_results")
time_str = "590s"
time_str2 = '590.0'
WavDir = 0
option_label = 'Option2'

# === APPLY OFFSETS TO PARAMETERS ===
APPLY_OFFSETS = True 
DOF_OFFSETS = {
    "telescoping": 5.0,
    "luffing": 0.0,
    "slewing": 0.0,
    "telescoping_vel": 0.0,
    "luffing_vel": 0.0,
    "slewing_vel": 0.0
}

# === FILE PATHS ===
prediction_dir = os.path.join(base_dir, "..", "prediction_results")
prediction_dir_Floater = os.path.join(base_dir, "..", "prediction_results", f"{option_label}_Floater_WD{WavDir}")
prediction_dir_SOV = os.path.join(base_dir, "..", "prediction_results", f"{option_label}_SOV_WD{WavDir}")

meas_sov_file = os.path.join(prediction_dir_SOV, f"measurements_{time_str2}_Option2_sov_WD{WavDir}.csv")
meas_floater_file = os.path.join(prediction_dir_Floater, f"measurements_{time_str2}_Option2_floater_WD{WavDir}.csv")

pred_sov_file = os.path.join(prediction_dir_SOV, f"PRED_HISTORY_{time_str}_Option2_sov_WD{WavDir}.csv")
pred_floater_file = os.path.join(prediction_dir_Floater, f"PRED_HISTORY_{time_str}_Option2_floater_WD{WavDir}.csv")

plot_subfolder = os.path.join(prediction_dir, f"{option_label}_WD{WavDir}")
os.makedirs(plot_subfolder, exist_ok=True)

# === LOAD DATA ===
df_meas_sov = pd.read_csv(meas_sov_file)
df_pred_sov = pd.read_csv(pred_sov_file)
df_meas_floater = pd.read_csv(meas_floater_file)
df_pred_floater = pd.read_csv(pred_floater_file)

# === GANGWAY MOTIONS ===
def compute_gangway(x_f, y_f, z_f, x_s, y_s, z_s):
    dist = np.sqrt((x_f - x_s)**2 + (y_f - y_s)**2 + (z_f - z_s)**2)
    tel = dist - dist.iloc[0]
    luf = np.degrees(np.arcsin(np.clip((z_f - z_s) / dist, -1.0, 1.0)))
    sle = np.degrees(np.arctan2(x_f - x_s, y_f - y_s))
    return tel, luf, sle

def compute_velocity_components(df_sov, df_floater, prefix=""):
    V = df_sov[[f"{prefix}Vx_SOV_local", f"{prefix}Vy_SOV_local", f"{prefix}Vz_SOV_local"]].values
    F = df_floater[[f"{prefix}Vx_Floater_local", f"{prefix}Vy_Floater_local", f"{prefix}Vz_Floater_local"]].values
    rel = V - F

    dx = df_sov[f"{prefix}PtfmTDX_SOV"] - df_floater[f"{prefix}PtfmTDX_Floater"]
    dy = df_sov[f"{prefix}PtfmTDY_SOV"] - df_floater[f"{prefix}PtfmTDY_Floater"]
    dz = df_sov[f"{prefix}PtfmTDZ_SOV"] - df_floater[f"{prefix}PtfmTDZ_Floater"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    ux, uy, uz = dx / dist, dy / dist, dz / dist

    telescoping_vel = np.sum(rel * np.column_stack([ux, uy, uz]), axis=1)
    luffing = np.degrees(np.arcsin(np.clip((df_floater[f"{prefix}PtfmTDZ_Floater"] - df_sov[f"{prefix}PtfmTDZ_SOV"]) / dist, -1.0, 1.0)))
    slewing = np.degrees(np.arctan2(df_floater[f"{prefix}PtfmTDX_Floater"] - df_sov[f"{prefix}PtfmTDX_SOV"],
                                    df_floater[f"{prefix}PtfmTDY_Floater"] - df_sov[f"{prefix}PtfmTDY_SOV"]))
    luffing_vel = np.gradient(luffing, df_sov["Time"])
    slewing_vel = np.gradient(slewing, df_sov["Time"])
    return telescoping_vel, luffing_vel, slewing_vel

# === THRESHOLDS ===
thresholds = {
    "telescoping": {"transfer": 4.0, "stay": 4.5, "unit": "[m]"},
    "luffing": {"transfer": 10.0, "stay": 15.0, "unit": "[deg]"},
    "slewing": {"transfer": 30.0, "stay": 50.0, "unit": "[deg]"},
    "telescoping_vel": {"transfer": 1.5, "stay": 2.5, "unit": "[m/s]"},
    "luffing_vel": {"transfer": 4.0, "stay": 6.0, "unit": "[deg/s]"},
    "slewing_vel": {"transfer": 4.0, "stay": 5.0, "unit": "[deg/s]"},
}

def plot_with_threshold_coloring(ax, time, series, thresholds, base_color, label_prefix, param_name):
    t = thresholds["transfer"]
    s = thresholds["stay"]
    normal = ~((series > t) | (series < -t))
    pt_exceed = ((series > t) | (series < -t)) & ~((series > s) | (series < -s))
    sc_exceed = (series > s) | (series < -s)

    def plot_segment(mask, color, label=None):
        if not mask.any():
            return
        idx = np.where(mask.values)[0]
        segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for seg in segments:
            ax.plot(time.iloc[seg], series.iloc[seg], color=color, label=label)
            label = None

    plot_segment(normal, base_color, f"{label_prefix} {param_name}")
    plot_segment(pt_exceed, "orange")
    plot_segment(sc_exceed, "red")

# === COMPUTE DOFs & VELOCITIES ===
tel_meas, luf_meas, sle_meas = compute_gangway(
    df_meas_floater["PtfmTDX_Floater"], df_meas_floater["PtfmTDY_Floater"], df_meas_floater["PtfmTDZ_Floater"],
    df_meas_sov["PtfmTDX_SOV"], df_meas_sov["PtfmTDY_SOV"], df_meas_sov["PtfmTDZ_SOV"]
)
tel_pred, luf_pred, sle_pred = compute_gangway(
    df_pred_floater["Predicted_PtfmTDX_Floater"], df_pred_floater["Predicted_PtfmTDY_Floater"], df_pred_floater["Predicted_PtfmTDZ_Floater"],
    df_pred_sov["Predicted_PtfmTDX_SOV"], df_pred_sov["Predicted_PtfmTDY_SOV"], df_pred_sov["Predicted_PtfmTDZ_SOV"]
)

tel_v_meas, luf_v_meas, sle_v_meas = compute_velocity_components(df_meas_sov, df_meas_floater)
tel_v_pred, luf_v_pred, sle_v_pred = compute_velocity_components(df_pred_sov, df_pred_floater, prefix="Predicted_")

df_meas = pd.DataFrame({
    "Time": df_meas_sov["Time"],
    "telescoping": tel_meas,
    "luffing": luf_meas,
    "slewing": sle_meas,
    "telescoping_vel": tel_v_meas,
    "luffing_vel": luf_v_meas,
    "slewing_vel": sle_v_meas
})
df_pred = pd.DataFrame({
    "Time": df_pred_sov["Time"],
    "telescoping": tel_pred,
    "luffing": luf_pred,
    "slewing": sle_pred,
    "telescoping_vel": tel_v_pred,
    "luffing_vel": luf_v_pred,
    "slewing_vel": sle_v_pred
})

# === TRIM TO COMMON TIME RANGE ===
t_min = max(df_meas["Time"].min(), df_pred["Time"].min())
t_max = min(df_meas["Time"].max(), df_pred["Time"].max())

df_meas = df_meas[(df_meas["Time"] >= t_min) & (df_meas["Time"] <= t_max)].reset_index(drop=True)
df_pred = df_pred[(df_pred["Time"] >= t_min) & (df_pred["Time"] <= t_max)].reset_index(drop=True)

# === PLOTTING ===
fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
xlim_min, xlim_max = df_pred["Time"].iloc[0], 600
params = ["telescoping", "luffing", "slewing", "telescoping_vel", "luffing_vel", "slewing_vel"]

for i, param in enumerate(params):
    ax = axs[i]
    plot_with_threshold_coloring(ax, df_meas["Time"], df_meas[param], thresholds[param], "blue", "Measured", param)

    pred_series = df_pred[param] + DOF_OFFSETS.get(param, 0.0) if APPLY_OFFSETS else df_pred[param]
    plot_with_threshold_coloring(ax, df_pred["Time"], pred_series, thresholds[param], "green", "Predicted", param)

    t = thresholds[param]["transfer"]
    s = thresholds[param]["stay"]
    ax.axhline(t, color="orange", linestyle="--", label=f"Personnel Transfer ±{t}")
    ax.axhline(-t, color="orange", linestyle="--")
    ax.axhline(s, color="red", linestyle="--", label=f"Stay Connected ±{s}")
    ax.axhline(-s, color="red", linestyle="--")

    pt_warning = ((pred_series < -t) | (pred_series > t)).any() or ((df_meas[param] < -t) | (df_meas[param] > t)).any()
    sc_warning = ((pred_series < -s) | (pred_series > s)).any() or ((df_meas[param] < -s) | (df_meas[param] > s)).any()
    pt_label = "PT: WARNING" if pt_warning else "PT: No Warning"
    sc_label = "SC: WARNING" if sc_warning else "SC: No Warning"

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color='none', label=pt_label), Line2D([], [], color='none', label=sc_label)]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax.set_ylabel(f"{param} {thresholds[param]['unit']}")
    ax.grid(True)
    ax.set_xlim(xlim_min, xlim_max)

axs[-1].set_xlabel("Time [s]")
fig.subplots_adjust(right=0.82)
fig.suptitle(f"Predicted Gangway Motions and Velocities {option_label}", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])

if APPLY_OFFSETS:
    for param in DOF_OFFSETS:
        if param in df_pred.columns:
            df_pred[param] += DOF_OFFSETS[param]

# === SAVE ===
plot_path = os.path.join(plot_subfolder, f"GangwayComparison_Option2_{time_str}_WD_{WavDir}deg.pdf")
df_meas.to_csv(os.path.join(plot_subfolder, f"MeasuredGangwayMotionsAndVelocities_Option2_{time_str}_WD_{WavDir}deg.csv"), index=False)
df_pred.to_csv(os.path.join(plot_subfolder, f"PredictedGangwayMotionsAndVelocities_Option2_{time_str}_WD_{WavDir}deg.csv"), index=False)

plt.savefig(plot_path, dpi=300)
plt.show()
print(f"[INFO] Files saved to: {plot_subfolder}")
