import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod.dolphinn import DOLPHINN as DOL
from Prediction_functions import save_prediction_csv, active_pred_plot

# Initialize the figure and axes
plt.ion()
fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

# Define the prediction_history DataFrame with default column names
prediction_history = pd.DataFrame(columns=['Time', 'Predicted_State'])  # DataFrame to store prediction history

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure, current_time, pred_error_x, pred_error_y, save_csv, save_csv_time, Prediction_state, sim_length, required_measurements, data_source, MLSTM_MODEL_NAME, WavDir, CaseNr):
    global prediction_history
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Use input data frame directly
    data = data_frame_inputs 
    present_time = round(data_frame_inputs["Time"].iloc[-1] - dol.time_horizon, 4)
    t1 = present_time
    t2 = dol.time_horizon
    t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
    t2_idx = np.where(np.min(np.abs(data['Time'] - (t2 + t1))) == np.abs(data['Time'] - (t2 + present_time)))[0][0]

    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time_data = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    t_pred, y_hat = dol.predict(time_data, state, wave, history=0)

    if plot_figure:
        active_pred_plot(t_pred, y_hat, pred_error_x, pred_error_y, data_frame_inputs, current_time, dol, time_data, t1_idx, t2, t1, fig, ax, Prediction_state)

    if current_time == save_csv_time and save_csv:
        save_prediction_csv(
        t_pred,
        y_hat,
        pred_error_x,
        pred_error_y,
        prediction_history,
        Prediction_state,
        save_csv_time,
        required_measurements,
        data_source,
        MLSTM_MODEL_NAME, 
        WavDir,
        CaseNr
    )
    
    history_data = pd.DataFrame({'Time': t_pred + t2})
    for param in required_measurements:
        history_data[f'Predicted_{param}'] = y_hat[param] + pred_error_y

    prediction_history = pd.concat([prediction_history, history_data], ignore_index=True)

    return t_pred, y_hat
