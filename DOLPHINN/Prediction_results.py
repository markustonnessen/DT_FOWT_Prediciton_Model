import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import re
from matplotlib.lines import Line2D


if len(sys.argv) == 3:
    measurement_filename = sys.argv[1]
    pred_history_filename = sys.argv[2]
    print(f"[INFO] Using passed-in files:")
else:
    raise ValueError("Must be called with measurement and prediction filenames as arguments.")

APPLY_OFFSETS = False
DOF_OFFSETS = {
    "telescoping": 1.0,
    "luffing": 0.0,
    "slewing": 0.0,
    "telescoping_vel": 0.0,
    "luffing_vel": 0.0,
    "slewing_vel": 0.0
}

def parse_subfolder_from_filename(filename):
    match_opt = re.search(r"_(Option\d+(?:_[A-Za-z]+)?)_", filename)
    match_wd = re.search(r"_WD(-?\d+)", filename) 
    option = match_opt.group(1) if match_opt else "Unknown"
    wavdir = match_wd.group(1) if match_wd else "0"
    return f"{option}_WD{wavdir}"

subfolder = parse_subfolder_from_filename(pred_history_filename)
print(f"[INFO] Detected subfolder: {subfolder}")

try:
    WavDir = int(subfolder.split("_WD")[1])
except (IndexError, ValueError):
    WavDir = 0

prediction_dir = os.path.join(os.path.dirname(__file__), "prediction_results", subfolder)

measurement_file = os.path.join(prediction_dir, measurement_filename)
pred_history_file = os.path.join(prediction_dir, pred_history_filename)

print(f"       Measurements: {os.path.basename(measurement_file)}")
print(f"       Predictions : {os.path.basename(pred_history_file)}")

for file_path in [measurement_file, pred_history_file]:
    for attempt in range(5):
        if os.path.exists(file_path):
            break
        time.sleep(0.2)
    else:
        raise FileNotFoundError(f"[ERROR] File not found after retries: {file_path}")

# === DETECT SOURCE TYPE AND TIMESTAMP FROM FILENAME ===
if "floater" in measurement_filename:
    sim_type = "Option2_Floater"
    option_label = "Option2_Floater"
elif "sov" in measurement_filename:
    sim_type = "Option2_SOV"
    option_label = "Option2_SOV"
elif "Option3" in measurement_filename:
    sim_type = "Option3"
    option_label = "Option3"
elif "Option1" in measurement_filename:
    sim_type = "Option1"
    option_label = "Option1"
else:
    sim_type = "Unknown"
    option_label = "Unknown"


match = re.search(r"measurements_(\d+)", measurement_filename)
save_time_str = match.group(1) if match else "unknown"

# === LOAD DATA ===
df_meas = pd.read_csv(measurement_file)
df_pred = pd.read_csv(pred_history_file)

# === MATCH PARAMETERS ===
meas_params = [col for col in df_meas.columns if col != "Time"]
pred_params = [col for col in df_pred.columns if col.startswith("Predicted_")]
shared_params = [p.replace("Predicted_", "") for p in pred_params if p.replace("Predicted_", "") in meas_params]

# === DETECT WHETHER GANGWAY STATES ARE PREDICTED ===
gangway_states = {"telescoping", "luffing", "slewing"}
predicted_gangway_states = set(p.replace("Predicted_", "") for p in pred_params)
predicts_gangway_directly = gangway_states.issubset(predicted_gangway_states)

# === DETECT WHETHER ALL DOFS FOR COMPUTING GANGWAY EXIST ===
required_dofs = [
    "Predicted_PtfmTDX_Floater", "Predicted_PtfmTDY_Floater", "Predicted_PtfmTDZ_Floater",
    "Predicted_PtfmTDX_SOV", "Predicted_PtfmTDY_SOV", "Predicted_PtfmTDZ_SOV",
    "Predicted_Vx_Floater_local", "Predicted_Vy_Floater_local", "Predicted_Vz_Floater_local",
    "Predicted_Vx_SOV_local", "Predicted_Vy_SOV_local", "Predicted_Vz_SOV_local"
]

can_compute_gangway = all(col in df_pred.columns for col in required_dofs)

param_units = {
    "slewing": " [deg]",
    "luffing": " [deg]",
    "telescoping": " [m]",
    "PtfmTDX_SOV": " [m]",
    "PtfmTDY_SOV": " [m]",
    "PtfmTDZ_SOV": " [m]",
    "PtfmRDX_SOV": " [deg]",
    "PtfmRDY_SOV": " [deg]",
    "PtfmRDZ_SOV": " [deg]",
    "PtfmTDX_Floater": " [m]",
    "PtfmTDY_Floater": " [m]",
    "PtfmTDZ_Floater": " [m]",
    "PtfmRDX_Floater": " [deg]",
    "PtfmRDY_Floater": " [deg]",
    "PtfmRDZ_Floater": " [deg]",
    "telescoping_vel": " [m/s]",
    "luffing_vel": " [deg/s]",
    "slewing_vel": " [deg/s]"
}

thresholds = {
    "telescoping":       {"transfer": 4.0, "stay": 4.5, "unit": "[m]"},
    "luffing":           {"transfer": 10.0, "stay": 15.0, "unit": "[deg]"},
    "slewing":           {"transfer": 30.0, "stay": 50.0, "unit": "[deg]"},
    "telescoping_vel":   {"transfer": 1.5, "stay": 2.5, "unit": "[m/s]"},
    "luffing_vel":       {"transfer": 4.0, "stay": 6.0, "unit": "[deg/s]"},
    "slewing_vel":       {"transfer": 4.0, "stay": 5.0, "unit": "[deg/s]"},
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


# === PLOT PREDICTED STATES ===
if sim_type == "Option3":
    gangway_dofs = ["telescoping", "luffing", "slewing", "telescoping_vel", "luffing_vel", "slewing_vel"]
    fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    xlim_min, xlim_max = df_pred["Time"].iloc[0], df_pred["Time"].iloc[-1]

    for i, param in enumerate(gangway_dofs):
        time_pred = df_pred["Time"]
        time_meas = df_meas["Time"]
        if APPLY_OFFSETS and param == "telescoping":
            y_pred = df_pred[f"Predicted_{param}"] + DOF_OFFSETS.get(param, 0.0)
        else:
            y_pred = df_pred[f"Predicted_{param}"]
        y_meas = df_meas[param]

        plot_with_threshold_coloring(axs[i], time_pred, y_pred, thresholds[param], "green", "Predicted", param)
        plot_with_threshold_coloring(axs[i], time_meas, y_meas, thresholds[param], "blue", "Measured", param)

        if param in thresholds:
            t = thresholds[param]["transfer"]
            s = thresholds[param]["stay"]

            axs[i].axhline(t, color="orange", linestyle="--", label=f"Personnel Transfer ±{t}")
            axs[i].axhline(-t, color="orange", linestyle="--")
            axs[i].axhline(s, color="red", linestyle="--", label=f"Stay Connected ±{s}")
            axs[i].axhline(-s, color="red", linestyle="--")

            idx_pt_pred = (y_pred < -t) | (y_pred > t)
            idx_sc_pred = (y_pred < -s) | (y_pred > s)

            idx_pt_meas = (y_meas < -t) | (y_meas > t)
            idx_sc_meas = (y_meas < -s) | (y_meas > s)

            pt_warning = idx_pt_pred.any() or idx_pt_meas.any()
            sc_warning = idx_sc_pred.any() or idx_sc_meas.any()

            pt_label = "PT: WARNING" if pt_warning else "PT: No Warning"
            sc_label = "SC: WARNING" if sc_warning else "SC: No Warning"

            handles, labels = axs[i].get_legend_handles_labels()
            handles += [Line2D([], [], color='none', label=pt_label),
                        Line2D([], [], color='none', label=sc_label)]
            axs[i].legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
            

        axs[i].set_ylabel(f"{param}{param_units.get(param, '')}")
        axs[i].grid(True)
        axs[i].set_xlim(xlim_min, xlim_max)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Predicted Gangway Motions and Velocities {option_label}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path1 = os.path.join(prediction_dir, f"GangwayComparison_{option_label}_{save_time_str}s_WD_{WavDir}deg.pdf")
    fig.savefig(plot_path1, dpi=300)
    plt.show()
    print(f"[INFO] Saved DOF comparison plot to: {plot_path1}")

else:
    fig, axs = plt.subplots(len(shared_params), 1, figsize=(12, 3 * len(shared_params)), sharex=True)
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    xlim_min, xlim_max = df_pred["Time"].iloc[0], df_pred["Time"].iloc[-1]

    for i, param in enumerate(shared_params):
        axs[i].plot(df_meas["Time"], df_meas[param], label=f"Measured {param}", color="blue")
        if APPLY_OFFSETS and param == "telescoping":
            y_pred_plot = df_pred[f"Predicted_{param}"] + DOF_OFFSETS.get(param, 0.0)
        else:
            y_pred_plot = df_pred[f"Predicted_{param}"]
        axs[i].plot(df_pred["Time"], y_pred_plot, label=f"Predicted {param}", color="green")
        axs[i].set_ylabel(f"{param}{param_units.get(param, '')}")
        axs[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        axs[i].grid(True)
        axs[i].set_xlim(xlim_min, xlim_max)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{option_label} Prediction vs. Measurements SIMA", fontsize=15)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path1 = os.path.join(prediction_dir, f"{option_label}_DOFs_vs_Measurement_{save_time_str}s_WD{WavDir}deg.pdf")
    fig.savefig(plot_path1, dpi=300)
    print(f"[INFO] Saved DOF comparison plot to: {plot_path1}")
    plt.show()


# =========================================== COMPUTE AND PLOT GANGWAY MOTIONS IF POSSIBLE ================================================
if can_compute_gangway:
    def compute_gangway(x_f, y_f, z_f, x_s, y_s, z_s):
        dist = np.sqrt((x_f - x_s)**2 + (y_f - y_s)**2 + (z_f - z_s)**2)
        tel = dist - dist.iloc[0]
        luf = np.degrees(np.arcsin(np.clip((z_f - z_s) / dist, -1.0, 1.0)))
        sle = np.degrees(np.arctan2(x_f - x_s, y_f - y_s))
        return tel, luf, sle

    tel_pred, luf_pred, sle_pred = compute_gangway(
        df_pred["Predicted_PtfmTDX_Floater"], df_pred["Predicted_PtfmTDY_Floater"], df_pred["Predicted_PtfmTDZ_Floater"],
        df_pred["Predicted_PtfmTDX_SOV"], df_pred["Predicted_PtfmTDY_SOV"], df_pred["Predicted_PtfmTDZ_SOV"]
    )

    df_gangway_pred = pd.DataFrame({
        "Time": df_pred["Time"],
        "telescoping": tel_pred,
        "luffing": luf_pred,
        "slewing": sle_pred
    })


    def compute_velocity_components(df, prefix):
        V = df[[f"{prefix}Vx_Floater_local", f"{prefix}Vy_Floater_local", f"{prefix}Vz_Floater_local"]].values
        S = df[[f"{prefix}Vx_SOV_local", f"{prefix}Vy_SOV_local", f"{prefix}Vz_SOV_local"]].values
        rel = S - V

        dx = df[f"{prefix}PtfmTDX_SOV"] - df[f"{prefix}PtfmTDX_Floater"]
        dy = df[f"{prefix}PtfmTDY_SOV"] - df[f"{prefix}PtfmTDY_Floater"]
        dz = df[f"{prefix}PtfmTDZ_SOV"] - df[f"{prefix}PtfmTDZ_Floater"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        ux, uy, uz = dx / dist, dy / dist, dz / dist

        telescoping_vel = np.sum(rel * np.column_stack([ux, uy, uz]), axis=1)

        luffing = np.degrees(np.arcsin(np.clip((df[f"{prefix}PtfmTDZ_Floater"] - df[f"{prefix}PtfmTDZ_SOV"]) / dist, -1.0, 1.0)))
        slewing = np.degrees(np.arctan2(df[f"{prefix}PtfmTDX_Floater"] - df[f"{prefix}PtfmTDX_SOV"],
                                        df[f"{prefix}PtfmTDY_Floater"] - df[f"{prefix}PtfmTDY_SOV"]))

        luffing_vel = np.gradient(luffing, df["Time"])
        slewing_vel = np.gradient(slewing, df["Time"])

        return telescoping_vel, luffing_vel, slewing_vel

    def check_thresholds(series, transfer, stay):
        """Return short exceedance status for PT and SC."""
        warn_pt = "Exceed" if ((series < -transfer) | (series > transfer)).any() else "No Warning"
        warn_sc = "Exceed" if ((series < -stay) | (series > stay)).any() else "No Warning"
        return warn_pt, warn_sc

    tel_v_pred, luf_v_pred, sle_v_pred = compute_velocity_components(df_pred, prefix="Predicted_")

    df_gangway_pred["telescoping_vel"] = tel_v_pred
    df_gangway_pred["luffing_vel"] = luf_v_pred
    df_gangway_pred["slewing_vel"] = sle_v_pred

    fig2, axs2 = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    try:
        tel_meas, luf_meas, sle_meas = compute_gangway(
            df_meas["PtfmTDX_Floater"], df_meas["PtfmTDY_Floater"], df_meas["PtfmTDZ_Floater"],
            df_meas["PtfmTDX_SOV"], df_meas["PtfmTDY_SOV"], df_meas["PtfmTDZ_SOV"]
        )

        df_gangway_meas = pd.DataFrame({
            "Time": df_meas["Time"],
            "telescoping": tel_meas,
            "luffing": luf_meas,
            "slewing": sle_meas
        })

        meas_time_start = df_gangway_pred["Time"].iloc[0]
        meas_time_end = df_gangway_pred["Time"].iloc[-1]
        df_gangway_meas = df_gangway_meas[
            (df_gangway_meas["Time"] >= meas_time_start) &
            (df_gangway_meas["Time"] <= meas_time_end)
        ].reset_index(drop=True)

        has_measured = True

        mask = (df_meas["Time"] >= meas_time_start) & (df_meas["Time"] <= meas_time_end)
        df_meas_trimmed = df_meas[mask].reset_index(drop=True)
        tel_v_meas, luf_v_meas, sle_v_meas = compute_velocity_components(df_meas_trimmed, prefix="")
        df_gangway_meas["telescoping_vel"] = tel_v_meas
        df_gangway_meas["luffing_vel"] = luf_v_meas
        df_gangway_meas["slewing_vel"] = sle_v_meas

    except KeyError:
        print("[WARNING] Missing measured DOFs - skipping measured gangway lines.")
        df_gangway_meas = pd.DataFrame()
        has_measured = False

    for i, motion in enumerate(["telescoping", "luffing", "slewing", "telescoping_vel", "luffing_vel", "slewing_vel"]):
        time_pred = df_gangway_pred["Time"]
        if APPLY_OFFSETS:
            y_pred = df_gangway_pred[motion] + DOF_OFFSETS.get(motion, 0.0)
        else:
            y_pred = df_gangway_pred[motion]

        plot_with_threshold_coloring(axs2[i], time_pred, y_pred, thresholds[motion], "green", "Predicted", motion)

        if motion in df_gangway_meas.columns:
            time_meas = df_gangway_meas["Time"]
            y_meas = df_gangway_meas[motion]
            plot_with_threshold_coloring(axs2[i], time_meas, y_meas, thresholds[motion], "blue", "Measured", motion)
            
        else:
            y_meas = pd.Series([], dtype=float)
            time_meas = pd.Series([], dtype=float)

        if motion in thresholds:
            t = thresholds[motion]["transfer"]
            s = thresholds[motion]["stay"]

            axs2[i].axhline(t, color="orange", linestyle="--", label=f"Personnel Transfer ±{t}")
            axs2[i].axhline(-t, color="orange", linestyle="--")
            axs2[i].axhline(s, color="red", linestyle="--", label=f"Stay Connected ±{s}")
            axs2[i].axhline(-s, color="red", linestyle="--")

            pt_warning = ((y_pred < -t) | (y_pred > t)).any() or ((y_meas < -t) | (y_meas > t)).any()
            sc_warning = ((y_pred < -s) | (y_pred > s)).any() or ((y_meas < -s) | (y_meas > s)).any()

            pt_label = "PT: WARNING" if pt_warning else "PT: No Warning"
            sc_label = "SC: WARNING" if sc_warning else "SC: No Warning"

            handles, labels = axs2[i].get_legend_handles_labels()
            handles += [Line2D([], [], color='none', label=pt_label),
                        Line2D([], [], color='none', label=sc_label)]
            axs2[i].legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
                         
        axs2[i].set_ylabel(f"{motion}{param_units.get(motion, '')}")
        axs2[i].grid(True)
        axs2[i].set_xlim(xlim_min, xlim_max)
    axs2[-1].set_xlabel("Time [s]")

    fig2.subplots_adjust(right=0.82)
    fig2.suptitle(f"Predicted Gangway Motions and Velocities {option_label}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path2 = os.path.join(prediction_dir, f"GangwayComparison_{option_label}_{save_time_str}s_WD_{WavDir}deg.pdf")
    plt.savefig(path2, dpi=300)

    df_gangway_pred.to_csv(os.path.join(prediction_dir, f"PredictedGangwayMotionsAndVelocities_{option_label}_{save_time_str}s_WD_{WavDir}deg.csv"), index=False)
    if has_measured:
        df_gangway_meas.to_csv(os.path.join(prediction_dir, f"MeasuredGangwayMotionsAndVelocities_{option_label}_{save_time_str}s_WD_{WavDir}deg.csv"), index=False)

    plt.show()
