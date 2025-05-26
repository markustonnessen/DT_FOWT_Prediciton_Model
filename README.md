# User Manual for Running the Digital Twin (DT) Framework

## Main Script: Run_DT_framework.py

This manual describes the different files and Python scripts developed and used by the authors within the DT framework. The complete repository is available at:
https://github.com/markustonnessen/DT_FOWT_Prediciton_Model

The MLSTM model utilized in this framework is obtained from the GitHub repository of Alkarem (2024).

The main driver file, `Run_DT_framework.py`, initiates the DT framework and defines the configuration for the SIMA model used for prediction. The script is located at:
https://github.com/markustonnessen/DT_FOWT_Prediciton_Model/blob/main/Launch_DT_framework.py

### SIMA Parameters

These parameters are used as input for the SIMA workflow simulation and can be adjusted by the user:

- `heading = 180`
- `xref = 25.875`
- `yref = -62.817`
- `hs = 3`
- `tp = 11`
- `xhook = 25.875`
- `yhook = -44.817`
- `SimLen = 2000`
- `WavDir = 0`

Descriptions:
- `heading`: Heading of the SOV in the simulation.
- `xref`, `yref`: Global position of the SOV.
- `hs`, `tp`: Significant wave height and peak period.
- `xhook`, `yhook`: Gangway hookup position.
- `SimLen`: Duration of the simulation.
- `WavDir`: Incoming wave direction.

### MLSTM and Prediction Parameters

These parameters configure the prediction model and its output:

- `MLSTM_MODEL_NAME = "Option2_LT_WD0_Floater"`
- `Prediction_state = "PtfmTDX_Floater"`
- `plot_figure = True`
- `timestep = 0.25`
- `time_horizon = 20.0`
- `pred_error_x = 0.0`
- `pred_error_y = 0.0`
- `pred_freq = 5.0`
- `save_csv = True`
- `save_csv_time = 590`
- `early_stop_enabled = True`
- `early_stop_time = 600`

Descriptions:
- Defines the model to use, which state to predict and how often to run predictions.
- `plot_figure`: If `True`, the predicted state will be plotted.
- `timestep`: Time resolution for data sampling.
- `time_horizon`: Duration of each prediction.
- `save_csv`: If `True`, results will be saved as `.csv`.

### Internal Paths and Files

Ensure that SIMA is installed, licensed, and the following paths are correctly set:

- `pathToSima = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation\sima-4.8.0-windows\sima.exe"`
- `pathToWorkspace = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation"`
- `pathToCommandFile = os.path.join(pathToWorkspace, "commandsModel.txt")`
- `args_path = os.path.join(pathToWorkspace, "sima_args.txt")`

Descriptions:
- `pathToSima`: Path to the SIMA executable.
- `pathToCommandFile`: Sends workflow parameters to SIMA.
- `args_path`: Used to pass prediction parameters.

---

## Additional Python Files in the Framework

- `Prediction_pipeline.py`: Determines which data to use based on the selected model.
- `Sima_output_converter.py`: Converts raw SIMA `.hdf5` outputs into `.csv` files.
- `MLSTM_batching.py`: Batches data for prediction; triggers prediction on condition.
- `MLSTM_predictor.py`: Triggers the `run_DOLPHINN()` function and generates plots.
- `Prediction_functions.py`: Helper functions for saving and plotting prediction results.
- `Prediction_results.py` / `Prediction_results_Option2.py`: Compare predictions with actual results and generate performance plots.

---

## SIMA Measurements

### From Floater

| Python Variable       | SIMA Variable              | Measurement Type         |
|-----------------------|----------------------------|--------------------------|
| wave_Floater          | Totalwaveelevation         | Wave Elevation           |
| PtfmTDX_Floater       | XGtranslationTotalmotion   | Surge                    |
| PtfmTDY_Floater       | YGtranslationTotalmotion   | Sway                     |
| PtfmTDZ_Floater       | ZGtranslationTotalmotion   | Heave                    |
| PtfmRDX_Floater       | XLrotationTotalmotion      | Roll                     |
| PtfmRDY_Floater       | XYrotationTotalmotion      | Pitch                    |
| PtfmRDZ_Floater       | ZGrotationTotalmotion      | Yaw                      |
| XLvelocity_Floater    | XLvelocityTotalmotion      | Surge Local Velocity     |
| YLvelocity_Floater    | YLvelocityTotalmotion      | Sway Local Velocity      |
| ZLvelocity_Floater    | ZLvelocityTotalmotion      | Heave Local Velocity     |
| XLang_vel_Floater     | XLang_vel_Totalmotion      | Roll Angular Velocity    |
| YLang_vel_Floater     | YLang_vel_Totalmotion      | Pitch Angular Velocity   |
| ZLang_vel_Floater     | ZLang_vel_Totalmotion      | Yaw Angular Velocity     |

### From SOV

| Python Variable       | SIMA Variable              | Measurement Type         |
|-----------------------|----------------------------|--------------------------|
| wave_SOV              | Totalwaveelevation         | Wave Elevation           |
| PtfmTDX_SOV           | XGtranslationTotalmotion   | Surge                    |
| PtfmTDY_SOV           | YGtranslationTotalmotion   | Sway                     |
| PtfmTDZ_SOV           | ZGtranslationTotalmotion   | Heave                    |
| PtfmRDX_SOV           | XLrotationTotalmotion      | Roll                     |
| PtfmRDY_SOV           | XYrotationTotalmotion      | Pitch                    |
| PtfmRDZ_SOV           | ZGrotationTotalmotion      | Yaw                      |
| XLvelocity_SOV        | XLvelocityTotalmotion      | Surge Local Velocity     |
| YLvelocity_SOV        | YLvelocityTotalmotion      | Sway Local Velocity      |
| ZLvelocity_SOV        | ZLvelocityTotalmotion      | Heave Local Velocity     |
| XLang_vel_SOV         | XLang_vel_Totalmotion      | Roll Angular Velocity    |
| YLang_vel_SOV         | YLang_vel_Totalmotion      | Pitch Angular Velocity   |
| ZLang_vel_SOV         | ZLang_vel_Totalmotion      | Yaw Angular Velocity     |