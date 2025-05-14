import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class PredictionClass():
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.batch_data = []                     # To store incoming data temporarily with a fixed max length        
        self.batch_size = 1000                   # Number of rows or timesteps per batch
        self.t_pred = pd.DataFrame()
        self.y_hat = pd.DataFrame()
        self.csv_saved = False                   # Flag to check if the CSV has been saved once
        self.initial_time = None
        self.timestep = 0.25
        self.data_frame_inputs = pd.DataFrame()  # Initialize DataFrame
        self.iteration_count = 0                 # Initialize the iteration count outside the loop
        self.full_measurements = []              # To store all measurements for all timesteps

    def run_simulation(self, current_time, measurements, plot_figure, time_horizon, pred_error_x, pred_error_y, pred_freq, save_csv, save_csv_time, FUTURE_WAVE_FILE, FOWT_pred_state, MLSTM_MODEL_NAME, DOLPHINN_PATH, sim_length, required_measurements, data_source, WavDir):
    
        from DOLPHINN.MLSTM_predictor import run_DOLPHINN

        self.iteration_count += 1  # Initialize the iteration count outside the loop

        if not hasattr(self, 'csv_df'):
            csv_file_path = os.path.join(os.path.dirname(__file__), "Incoming_Waves", FUTURE_WAVE_FILE)
            
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"[ERROR] Wave CSV not found: {csv_file_path}")

            self.csv_df = pd.read_csv(csv_file_path)

        # Set initial time if it has not been set
        if self.initial_time is None:
            self.initial_time = current_time
        desired_time = current_time + time_horizon

        matching_rows = self.csv_df[np.isclose(self.csv_df['Time'], desired_time)]
        if matching_rows.empty:
            print(f"[WARNING] No wave data found for desired_time={desired_time}. Skipping this timestep.")
            return None, None
        wave_measurement = matching_rows['wave'].iloc[0]

        # Extract measurement values for this timestep
        measurement_values = [measurements.get(key, 0.0) for key in required_measurements]

        # Append directly with real values (no None placeholders)
        self.batch_data.append([current_time, wave_measurement] + measurement_values)
        self.full_measurements.append([current_time] + measurement_values)


        if current_time >= time_horizon + self.initial_time and len(self.batch_data) <= self.batch_size:
            future_index = (current_time - self.initial_time) / self.timestep
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(future_index - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values

        if len(self.batch_data) > self.batch_size:
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(self.batch_size - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values
            popped_row = self.batch_data.pop(0)

        if self.iteration_count % 200 == 0 and len(self.batch_data) < self.batch_size:
            print(f"Remaining rows until initializing MLSTM for prediction: {self.batch_size - len(self.batch_data)} (Batch size: {len(self.batch_data)})")
            
        # Check if the last time value is at a whole second
        if len(self.batch_data) >= self.batch_size and current_time % pred_freq == 0:
            data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time', 'wave'] + required_measurements)
            print("Running MLSTM with input data frame shape:", data_frame_inputs.shape)
            #self.t_pred, self.y_hat = run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure, current_time, pred_error_x, pred_error_y, save_csv, save_csv_time, FOWT_pred_state, sim_length, required_measurements)
            self.t_pred, self.y_hat = run_DOLPHINN(
            data_frame_inputs, DOLPHINN_PATH, plot_figure, current_time,
            pred_error_x, pred_error_y, save_csv, save_csv_time,
            FOWT_pred_state, sim_length, required_measurements, data_source, MLSTM_MODEL_NAME, WavDir
            )
            if save_csv and not self.csv_saved and current_time >= save_csv_time:
                self.csv_saved = True

                self.control_csv_saved = True
                # Save only the required_measurements along with the Time column
                output_file_path = os.path.join(self.this_dir, "Control_Data", f"Control_Data_T{current_time}_active.csv")

                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                data_frame_inputs.to_csv(output_file_path, index=False)
                print(f"SAVED control CSV at t = {current_time}")

        if current_time == save_csv_time and save_csv:

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

            subfolder_name = f"{option_label}_WD{WavDir}"

            # Build full save path
            base_dir = os.path.dirname(__file__)
            prediction_results_dir = os.path.join(base_dir, "prediction_results", subfolder_name)
            os.makedirs(prediction_results_dir, exist_ok=True)
            
            full_measurements_df = pd.DataFrame(self.full_measurements, columns=['Time'] + required_measurements)
            source_tag = data_source.capitalize()
            full_output_file_path = os.path.join(prediction_results_dir, f"measurements_{current_time}_{source_tag}_WD{WavDir}.csv")
            full_measurements_df.to_csv(full_output_file_path, index=False)
            print(f"[INFO] SAVED measurements at t = {current_time} to: {full_output_file_path}")

            self.data_frame_inputs = data_frame_inputs  # Ensure data_frame_inputs is assigned

        if hasattr(self, 'y_hat') and not self.y_hat.empty:
            return self.y_hat, self.t_pred
        else:
            return None, None
