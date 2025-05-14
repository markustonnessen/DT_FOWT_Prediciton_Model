import sys
import os
import yaml
import pandas as pd
import subprocess
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DOLPHINN.MLSTM_batching import PredictionClass
from DOLPHINN.Sima_output_converter import extract_and_save_csv

# -------------------- READ ARGS FROM FILE IF EXISTS --------------------
params_file = os.path.join(os.path.dirname(__file__), "SIMA_simulation", "sima_args.txt")
if os.path.exists(params_file):
    with open(params_file, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                try:
                    val = eval(val)
                    globals()[key] = val
                except Exception as e:
                    print(f"[WARNING] Failed to evaluate {key}={val}: {e}")

# ------------------- FUNCTION TO GET REQUIRED MEASUREMENTS --------------------
def get_model_config(model_name):
    base_dir = os.path.dirname(__file__)  # Absolute path to DOLPHINN/
    config_path = os.path.join(base_dir, "saved_models", model_name, "wave_model", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config.yaml not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "dof" not in config or not config["dof"]:
        raise ValueError(f"[ERROR] 'dof' field missing or empty in config.yaml at: {config_path}")
    
    training_path = config.get("training_dataset", "")
    if "Floater" in training_path:
        source = "Option2_Floater"
    elif "SOV" in training_path:
        source = "Option2_SOV"
    elif "Option3" in training_path:
        source = "Option3"
    elif "Option1" in training_path:
        source = "Option1"
    else:
        raise ValueError(f"[ERROR] Unknown data source in training_dataset path: {training_path}")

    return config["dof"], source

# -------------------- DEFAULT FALLBACKS --------------------
MLSTM_MODEL_NAME = globals().get("MLSTM_MODEL_NAME", "Con_set_TS_025_1")
timestep = globals().get("timestep", 0.25)
plot_figure = globals().get("plot_figure", True)
time_horizon = globals().get("time_horizon", 20.0)
pred_error_x = globals().get("pred_error_y", 0.0)
pred_error_y = globals().get("pred_error_x", 0.0)
pred_freq = globals().get("pred_freq", 5.0)
save_csv = globals().get("save_csv", True)
save_csv_time = globals().get("save_csv_time", 400)
FOWT_pred_state = globals().get("FOWT_pred_state", "PtfmTDX_Floater")
early_stop_enabled = globals().get("early_stop_enabled", False)
early_stop_time = globals().get("early_stop_time", 600)
WavDir = globals().get("WavDir", 0)  # Default to 0 if not set

# Dynamically load DOFs used for training from YAML
required_measurements, data_source = get_model_config(MLSTM_MODEL_NAME)
print(f"[INFO] Using DOFs from config.yaml: {required_measurements}")
print(f"[INFO] Detected data source from training_dataset path: {data_source}")

# -------------- PATH TO TRAINED MODEL ---------------
DOLPHINN_PATH = os.path.join(os.path.dirname(__file__), "saved_models", MLSTM_MODEL_NAME, "wave_model")

# ------------------- PREPROCESS ---------------------
(
    df_motion_Floater,
    df_motion_SOV,
    wave_path_Floater,
    motion_path_Floater,
    wave_path_SOV,
    motion_path_SOV,
    sim_length,
    df_motion_Gangway,
    motion_path_Gangway,
    wave_path_Gangway,
    df_motion_Option1,
    motion_path_Option1,
    wave_path_Option1
) = extract_and_save_csv(sys.argv, timestep)


# ------------------- SELECT SOV OR FLOATER ------------------------
if data_source == "Option2_Floater":
    df_motion = df_motion_Floater
    wave_file_path = wave_path_Floater
    motion_file_path = motion_path_Floater
    print(f"[INFO] Using Option2_Floater data for prediction.")
elif data_source == "Option2_SOV":
    df_motion = df_motion_SOV
    wave_file_path = wave_path_SOV
    motion_file_path = motion_path_SOV
    print(f"[INFO] Using Option2_SOV data for prediction.")
elif data_source == "Option3":
    df_motion = df_motion_Gangway
    wave_file_path = wave_path_Gangway
    motion_file_path = motion_path_Gangway
    print(f"[INFO] Using Option3 data for prediction.")
elif data_source == "Option1":
    df_motion = df_motion_Option1
    wave_file_path = wave_path_Option1
    motion_file_path = motion_path_Option1
    print(f"[INFO] Using Option1 data for prediction.")

else:
    raise ValueError(f"[ERROR] Unknown data source: {data_source}")

# ------------------- PREDICT ------------------------
pc = PredictionClass()
print("\n[INFO] Starting prediction...\n")

for _, row in df_motion.iterrows():
    current_time = row["Time"]
    if early_stop_enabled and current_time >= early_stop_time:
        print(f"[INFO] Early stopping triggered at {current_time}s (Limit: {early_stop_time}s)")
        break

    measurements = row.to_dict()
    print("[DEBUG] CSV columns:", df_motion.columns.tolist())
    print("[DEBUG] Required measurements:", required_measurements)

    pc.run_simulation(
        current_time=current_time,
        measurements=measurements,
        plot_figure=plot_figure,
        time_horizon=time_horizon,
        pred_error_x=pred_error_x,
        pred_error_y=pred_error_y,
        pred_freq=pred_freq,
        save_csv=save_csv,
        save_csv_time=save_csv_time,
        FUTURE_WAVE_FILE=wave_file_path,
        FOWT_pred_state=FOWT_pred_state,
        MLSTM_MODEL_NAME=MLSTM_MODEL_NAME,
        DOLPHINN_PATH=DOLPHINN_PATH,
        sim_length=sim_length,
        required_measurements=required_measurements,
        data_source=data_source,
        WavDir = WavDir
    )

print("\n[INFO] Done! Predictions have been generated.\n")

# ------------------- AFTER PREDICTIONS ------------------------
# Determine filenames based on known save time and data source
sim_type = data_source.capitalize()
save_time_str_decimal = f"{save_csv_time:.1f}"   # "590.0"
save_time_str_int = str(int(save_csv_time))      # "590"

measurement_file = f"measurements_{save_time_str_decimal}_{sim_type}_WD{WavDir}.csv"
pred_history_file = f"PRED_HISTORY_{save_time_str_int}s_{sim_type}_WD{WavDir}.csv"

plot_script_path = os.path.join(os.path.dirname(__file__), "Prediction_results.py")
if os.path.exists(plot_script_path):
    print("[INFO] Launching plot script for latest prediction results...")
    time.sleep(1)  # Optional: wait 1 second, only if needed
    subprocess.call(["python", plot_script_path, measurement_file, pred_history_file])
else:
    print("[WARNING] Plot script not found at:", plot_script_path)