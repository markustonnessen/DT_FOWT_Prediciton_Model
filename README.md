# User Manual for Running the Digital Twin (DT) Framework

## Main Script: Run_DT_framework.py

This user manual represents the different files and codes developed and used by the authors, within the DT framework. The MLSTM model utilized in this framework are obtained from the GitHub repository of Alkarem, at https://github.com/Yuksel-Rudy/DOLPHINN.git. 
The main driver file, `Launch_DT_framework.py`, initiates the DT framework and defines the configuration for the SIMA model used for prediction. The `Launch_DT_framework.py` script is located at https://github.com/markustonnessen/DT_FOWT_Prediciton_Model/blob/main/Launch_DT_framework.py and the configurations within this script are explained below. 

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


These parameters can be adjusted for the SIMA workflow simulation. The complete SIMA model is a .stask file, with various models and parameters which are adjustable. These parameters do not need manually adjusting, and are saved in the .stask. 

heading refers to the heading of the SOV in the SIMA workflow simulation. 
    
xref and yref refer to the global position of the SOV in x- and y-direction. 
    
hs and tp are the significant wave height and peak period, used for the simulated sea state. 
    
xhook and yhook refer to the global hookup position for the gangway, modeled as a point, in the x- and y-directions. 
    
SimLen refers to the simulation length for the SIMA workflow.
    
WavDir refers to the incoming wave direction in the simulation.

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


The trained model gets a `MLSTM_MODEL_NAME`. By setting this to `"Option2_LT_WD0_Floater"` the trained model for Option2, long-term sea states and wave direction of 0 deg. 

`Prediction_state` refers to the desired state to plot and showcase during prediction. After the prediction is done, all predicted system states are highlighted in plots. Only one system state is showcased during prediction, as showcasing them all would need great computational power. 

`plot_figure` acts as a toggle button to control whether the plot with the desired `Prediction_state` shall be generated or not.

`timestep` is the base time resolution of the simulation and prediction data. It defines how often the data points are sampled in the measured input signals and predicted output sequences. 

`time_horizon` sets how far into the future, in seconds, each prediction extends, for each `timestep`.

`pred_error_x` and `pred_error_y` are manual constant offsets applied to align predicted data better in relation to the measured data, in the x- and y-axis. Used as desired.

`pred_freq` refers to how often the prediction model is triggered to make a new prediction, in seconds. This parameter is adjustable for the user and depends on the computational efficiency desired.  

`save_csv` acts as a toggle button to control whether or not a .csv file with the prediction history shall be saved.

`save_csv_time` refers to when the prediction history .csv file shall be saved, and includes all previous prediction data from when the prediction started.

`early_stop_enabled` acts as a toggle button to enable an early stopping of the prediction model, even tho the SIMA simulation extends for a longer duration. This is useful in scenarios where predictions for the whole SIMA simulation aren't necessary to generate.

`early_stop_time` refers to when, in time, the prediction process shall stop if `early_stop_enabled = True`. If this is set to `early_stop_enabled = False`, the prediction process runs throughout the whole process. 

### Internal Paths and Files

In order to be able to run the DT framework and SIMA model, SIMA has to be downloaded, used with a valid license, and the system needs to know where to initiate the software. It also needs to make additional files for the framework to function, as the framework utilizes these files to know which states to use for prediction and data collection.

- `pathToSima = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation\sima-4.8.0-windows\sima.exe"`
- `pathToWorkspace = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation"`
- `pathToCommandFile = os.path.join(pathToWorkspace, "commandsModel.txt")`
- `args_path = os.path.join(pathToWorkspace, "sima_args.txt")`


`pathToSima` refers to where the `sima.exe` file is located, in order to execute the SIMA workflow simulation.

`pathToWorkspace` refers to where the `SIMA_simulation` folder is located. Within the GitHub repository, this folder is located at the correct location, and the user do not need to adjust this path if the repository is used.

`pathToCommandFile` is the path to the commandsModel.txt, which sends the desired manually set SIMA workflow parameters to the SIMA software before the simulation starts. Within this file, the path to the desired .stask model file needs to be updated. 

`args_path` is the path to the `sima_args.txt`, which sends the desired MLSTM and prediction parameters to the system. The `Prediction_pipeline.py` utilizes this file to send data where it is supposed to be. 


## Additional Python Files in the Framework

- `Prediction_pipeline.py`: This file determines what data from the SIMA workflow simulation to use. The framework extracts data and this file determine what parts of the data to use for prediction, based on the `MLSTM_MODEL_NAME` chosen. It then loads the trained models `config.yaml` file, determining which data source to be utilized, depending on which dataset the trained model used for training; `Option1`, `Option2_Floater`, `Option2_SOV` or `Option3`.
- `Sima_output_converter.py`: This file is responsible for reading the raw `HDF5` output file from the SIMA workflow simulation, converting the relevant data determined by `Prediction_pipeline.py` into `.csv` files. It separates wave elevations and motions/velocities or calculates gangway motions based on the desired data source. These files are further on used as input for the prediction.
- `MLSTM_batching.py`: This file defines the PredictonClass, which collects simulation data from `Sima_output_converter.py` and builds batches of measured data used for prediction. The desired batch size is determined within this file, with `self.batch_size = 1000` for example. When `len(self.batch_data) >= batch_size` and `current_time \% pred_freq == 0` the prediction is triggered.
- `MLSTM_predictor.py`: Ones a full batch is ready from `MLSTM_batching.py`, this file is triggered. Everytime a new prediction is due, the `run_DOLPHINN()` function gets triggered, and this script are responsible for making the single `Prediction_state` plot ready to be visualized and making `.csv` of prediction data ready for saving within the `Prediction_functions.py`.
- `Prediction_functions.py`: This file provides functions which supports the DOLPHINN system, mostly for saving prediction results into `.csvs` and visualizing live predictions for the chosen `Prediction_state`. 
- `Prediction_results.py` / `Prediction_results_Option2.py`: These files are responsible for comparing and visualizing the prediction results against the measured data from the SIMA workflow simulation. These files load the stored prediction history and measurements `.csvs` and calculate or generate gangway plots directly, depending on the `MLSTM_MODEL_NAME` and option chosen for training. Gangway thresholds are highlighted to obtain comparable results.

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


### Step by step guide to utilize the framework

1. Training of the MLSM model:
    Open `wave.yaml` within "dol_input" folder. In this file the user has the ability to set the desired states wanted for training, along with their units and desired training parameters. The framework relies on training datasets, following the desired `training_dataset` path within `wave.yaml`. This dataset needs to include `Option1, 2 or 3`. If `Option2` is used for training, the path need to include either `Floater` or `SOV` as well to work. These statements are determined within `Prediction_pipeline.py`, and dynamically makes the framework understand what it shall predict for. By adjusting `Prediction_pipeline.py`, and additional files, this dynamic interaction can be changed. 
    Open 01_wave_train.py in "examples" folder.`TEST = "Option3_LT_WD0"` refers to the name the saved trained model is saved as after training. This saved model can be located within "training_results", ""Option3_LT_WD0", "wave_Option1_LT_WD0.pdf".
    After training, the results are stored within "training_results" folder. The folder structure within this folder is dependent on what the "TEST" variable in `01_wave_train.py` is set as.

2. Ensure SIMA is downloaded and a valid license is available.
    In order for this framework to run, an Workflow model needs to be developed within SIMA, running the desired model (.stask). Within this workflow, a external program block needs to be implemented and be pointed to relevant file paths to execte Python.

3. Prediction:
    Open `Launch_DT_framework`. Within this file multiple parameters can be adjusted for the SIMA simulation, and all relevant file paths needs to be updated as mentioned over.
    The `MLSTM_MODEL_NAME = "Option3_LT_WD0"` needs to be set to a valid trained model folder name. The `Prediction_state = "PtfmTDZ_SOV"` needs to be set to a valid state used on training, from `wave.yaml`.


## Acknowledgements

The project has utilizd codes fromt eh following repositories:

- `DOLPHINN`: An MLSTM model has been developed by Yuksel R. Alkarem for predicting behaviors based on incoming wave data in offshore environments, at the GitHub repositroy https://github.com/Yuksel-Rudy/DOLPHINN.git
- `FOWT_Digital_Twin_ROSCO`: Previous work done by Fredrik Selheim Flesland and Hans Petter Tharaldsen has been utilized in this framework. The files `data_batching.py`, `wave_predict.py` and `prediction_functions.py` has been utilized and adjusted to better suit this framewrok. Their repository is located at https://github.com/HPtharaldsen/FOWT_Digital_Twin_ROSCO.git
- `NUWind Project`: This project has been carried it in a collaboration between the Norwegian University of Science and Technology (NTNU) and the University of Maine (UMaine). The students have gotten the opportunity to collabarate with each other and carry out research within the offshore wind field. 
