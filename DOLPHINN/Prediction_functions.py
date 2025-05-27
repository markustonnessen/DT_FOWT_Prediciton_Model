import sys
from pathlib import Path
import os
import math
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_prediction_csv(t_pred, y_hat, pred_error_x, pred_error_y, prediction_history, Prediction_state, save_time, required_measurements, data_source, MLSTM_MODEL_NAME, WavDir):
    print("Saving results to csv")

    # Round sim length for filename
    save_time_str = str(int(save_time))
    source_tag = data_source.capitalize()

    base_dir = os.path.dirname(__file__)
    wavdir = WavDir

    if "Option1" in MLSTM_MODEL_NAME:
        option_label = "Option1"
    elif "Floater" in MLSTM_MODEL_NAME:
        option_label = "Option2_Floater"
    elif "SOV" in MLSTM_MODEL_NAME:
        option_label = "Option2_SOV"
    elif "Option3" in MLSTM_MODEL_NAME:
        option_label = "Option3"
    else:
        option_label = "Unknown"

    subfolder_name = f"{option_label}_WD{wavdir}"

    # Build full save path
    base_dir = os.path.dirname(__file__)
    prediction_results_dir = os.path.join(base_dir, "prediction_results", subfolder_name)
    os.makedirs(prediction_results_dir, exist_ok=True)

    # Dynamic filenames
    # prediction_results_path = os.path.join(prediction_results_dir, f"PREDICTION_{save_time_str}s_{source_tag}_ACTIVE.csv")
    prediction_history_path = os.path.join(prediction_results_dir, f"PRED_HISTORY_{save_time_str}s_{source_tag}_WD{WavDir}.csv")

    # Define DOFs in degrees
    angle_dofs = [
        "PtfmRDX_Floater", "PtfmRDY_Floater", "PtfmRDZ_Floater",
        "PtfmRDX_SOV", "PtfmRDY_SOV", "PtfmRDZ_SOV",
        "luffing", "slewing"
    ]

    # Save prediction results
    prediction_results = pd.DataFrame({'Time': t_pred})
    for param in required_measurements:
        if param in angle_dofs:
            prediction_results[f'True_{param}'] = y_hat[param] + pred_error_y
        else:
            prediction_results[f'True_{param}'] = y_hat[param] + pred_error_x

    #prediction_results.to_csv(prediction_results_path, index=False)

    # Save prediction history
    prediction_history.to_csv(prediction_history_path, index=False)

    #print(f"[INFO] Saved prediction results to: {prediction_results_path}")
    print(f"[INFO] Saved prediction history to: {prediction_history_path}")

def active_pred_plot(t_pred, y_hat, pred_error_x, pred_error_y, data_frame_inputs, current_time, dol, time_data, t1_idx, t2, t1, fig, ax, Prediction_state, save_final_plot=False):

    # Initialize or update the plot elements
    if not hasattr(active_pred_plot, 'initialized'):
        active_pred_plot.line_actual, = ax.plot([], [], color='blue', label=f'Measured {Prediction_state} (SIMA)')
        active_pred_plot.line_predicted, = ax.plot([], [], color='#3CB371', linestyle='-', label=f'Predicted {Prediction_state}')
        active_pred_plot.line_history, = ax.plot([], [], color='#3CB371', linestyle="--", label='Prediction history')
        active_pred_plot.marker_actual = ax.scatter([], [], color='blue', alpha=0.5)
        active_pred_plot.marker_predicted = ax.scatter([], [], color='#3CB371', alpha=0.5)
        active_pred_plot.old_predictions = []  # List to store old prediction lines
        active_pred_plot.plotted_times = set()  # Set to store times for which history has been plotted
        active_pred_plot.last_stippled_time = None  # Track the last time a stippled line was added
        active_pred_plot.initialized = True

    # Convert current predicted line to stippled and add to old predictions if sufficient time has passed
    if active_pred_plot.last_stippled_time is None or current_time - active_pred_plot.last_stippled_time >= 10:  # adjust interval as needed
        if active_pred_plot.line_predicted.get_xdata().size > 0:
            old_line = ax.plot(active_pred_plot.line_predicted.get_xdata(), active_pred_plot.line_predicted.get_ydata(), linestyle="--", color='#3CB371')[0]
            active_pred_plot.old_predictions.append(old_line)
        active_pred_plot.last_stippled_time = current_time

    # Clear current prediction line data
    active_pred_plot.line_predicted.set_data([], [])

    # Set new data for current prediction
    active_pred_plot.line_predicted.set_data(t_pred + t2 + pred_error_x, y_hat[f"{Prediction_state}"] + pred_error_y)
    if Prediction_state in ["PtfmRDX_Floater", "PtfmRDY_Floater", "PtfmRDZ_Floater"]:
        actual_values = data_frame_inputs[f"{Prediction_state}"].iloc[:t1_idx]
        last_actual_state = data_frame_inputs[f"{Prediction_state}"].iloc[t1_idx - 1]
    else:
        actual_values = data_frame_inputs[f"{Prediction_state}"].iloc[:t1_idx]
        last_actual_state = data_frame_inputs[f"{Prediction_state}"].iloc[t1_idx - 1]

    active_pred_plot.line_actual.set_data(time_data.iloc[0:t1_idx] + t2, actual_values)
    active_pred_plot.marker_actual.set_offsets((time_data.iloc[t1_idx - 1] + t2, last_actual_state))

    
    # Update marker and text for actual data
    last_actual_time = time_data.iloc[t1_idx-1] + t2
    last_actual_state = data_frame_inputs[f"{Prediction_state}"].iloc[t1_idx-1] * 180/np.pi
    active_pred_plot.marker_actual.set_offsets((last_actual_time, last_actual_state))
    active_pred_plot.marker_actual.set_label(f'Current {Prediction_state} ({current_time}s)')

    # Update marker and text for predicted data
    last_pred_time = t_pred.iloc[-1] + t2 + pred_error_x
    last_pred_state = y_hat[f"{Prediction_state}"].iloc[-1] + pred_error_y
    active_pred_plot.marker_predicted.set_offsets((last_pred_time, last_pred_state))
    active_pred_plot.marker_predicted.set_label(f'Predicted {Prediction_state} ({current_time + dol.time_horizon + pred_error_x}s)')  

    # === UNIT-AWARE Y-LABELING ===
    deg_params = [
        "PtfmRDX_Floater", "PtfmRDY_Floater", "PtfmRDZ_Floater",
        "PtfmRDX_SOV", "PtfmRDY_SOV", "PtfmRDZ_SOV",
        "luffing", "slewing"
    ]
    deg_per_s_params = [
        "Wx_Floater", "Wy_Floater", "Wz_Floater",
        "Wx_SOV", "Wy_SOV", "Wz_SOV",
        "luffing_vel", "slewing_vel"
    ]
    m_per_s_params = [
        "Vx_Floater_local", "Vy_Floater_local", "Vz_Floater_local",
        "Vx_SOV_local", "Vy_SOV_local", "Vz_SOV_local",
        "telescoping_vel"
    ]

    if Prediction_state in deg_params:
        y_label = f"{Prediction_state} [deg]"
    elif Prediction_state in deg_per_s_params:
        y_label = f"{Prediction_state} [deg/s]"
    elif Prediction_state in m_per_s_params:
        y_label = f"{Prediction_state} [m/s]"
    else:
        y_label = f"{Prediction_state} [m]"

    # Combine actual and predicted values for range calculation
    all_values = np.concatenate([actual_values.values, y_hat[f"{Prediction_state}"].values + pred_error_y])

    # Compute min and max with margin
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    margin = 0.05 * (y_max - y_min) if y_max != y_min else 0.1

    ax.set_xlim((t1 - 50, t1 + 50))
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.grid(True)
    plt.title(f'{Prediction_state} Prediction. Wave Time Horizon: {dol.time_horizon}s')
    plt.draw()
    plt.pause(0.1)

    if save_final_plot:
        base_dir = os.path.dirname(__file__)
        figures_dir = os.path.join(base_dir, "prediction_results")
        os.makedirs(figures_dir, exist_ok=True)

        file_name = f"{Prediction_state}_final_prediction.png"
        file_path = os.path.join(figures_dir, file_name)

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved final prediction plot to: {file_path}")

    plt.show()


