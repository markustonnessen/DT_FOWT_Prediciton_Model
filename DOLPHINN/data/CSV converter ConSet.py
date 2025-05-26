import h5py
import pandas as pd
import numpy as np
import os
from natsort import natsorted

# === CONFIGURATION ===
input_h5_path = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\data\WaveDir_0.h5"
output_csv_path = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\data\WaveDir_0.csv"

# === CHOOSE WHICH CONDITION SETS FROM RAN SIMA SIMULATION, TO INCLUDE ===
condition_range = (1, 4)

# === OPEN HDF5 FILE ===
hf = h5py.File(input_h5_path, 'r')

combined_gangway_data = pd.DataFrame()
time_step = 0.25
current_time_offset = 0.0
condition_data = {}
max_time = 7500
num_steps = int(max_time / time_step)
time_array = np.linspace(0, max_time, num_steps)

condition_sets = list(hf['IEA15MW_VolturnUS_S1/ConditionSet'].keys())
condition_sets = [cond for cond in condition_sets if cond != "Variables"]
condition_sets = natsorted(condition_sets)

if condition_range is not None:
    start, end = condition_range
    condition_sets = [f"ConditionSet_{i}" for i in range(start, end + 1)]

def euler_to_rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform_velocity(v_A, omega, r_BA):
    """
    Rigid body velocity transformation.
    v_A: velocity at reference point A (np.array shape (3,))
    omega: angular velocity (np.array shape (3,))
    r_BA: position vector from A to B (np.array with shape (3,1))
    """
    omega_cross = np.cross(omega, r_BA.flatten())
    return v_A + omega_cross


for condition in condition_sets:
    if condition == "Variables":
        continue
    condition_path = f"IEA15MW_VolturnUS_S1/ConditionSet/{condition}/Dynamic"
    path_sov = f"{condition_path}/SOV/Global total position"
    path_floater = f"{condition_path}/Floater/Global total position"
    path_wave = f"{condition_path}"
    path_velo_sov  = f"{condition_path}/SOV/Local velocity"
    path_velo_floater = f"{condition_path}/Floater/Local velocity"
    data = {}
    try:
        XGtranslationTotalmotion_floater = hf[f'{path_floater}/XGtranslationTotalmotion'][3000:]
        YGtranslationTotalmotion_floater = hf[f'{path_floater}/YGtranslationTotalmotion'][3000:]
        ZGtranslationTotalmotion_floater = hf[f'{path_floater}/ZGtranslationTotalmotion'][3000:]
        XLrotationTotalmotion_floater = np.deg2rad(hf[f'{path_floater}/XLrotationTotalmotion'][3000:])
        YLrotationTotalmotion_floater = np.deg2rad(hf[f'{path_floater}/YLrotationTotalmotion'][3000:])
        ZGrotationTotalmotion_floater = np.deg2rad(hf[f'{path_floater}/ZGrotationTotalmotion'][3000:])

        XLvelocity_floater = hf[f'{path_velo_floater}/XLvelocityTotalmotion'][3000:]
        YLvelocity_floater = hf[f'{path_velo_floater}/YLvelocityTotalmotion'][3000:]
        ZLvelocity_floater = hf[f'{path_velo_floater}/ZLvelocityTotalmotion'][3000:]
        XLang_vel_floater = np.deg2rad(hf[f'{path_velo_floater}/XLang_vel_Totalmotion'][3000:])
        YLang_vel_floater = np.deg2rad(hf[f'{path_velo_floater}/YLang_vel_Totalmotion'][3000:])
        ZLang_vel_floater = np.deg2rad(hf[f'{path_velo_floater}/ZLang_vel_Totalmotion'][3000:])

        wave_Floater = hf[f'{path_wave}/Floater/Wave elevation/Totalwaveelevation'][3000:]


        XGtranslationTotalmotion_sov = hf[f'{path_sov}/XGtranslationTotalmotion'][3000:]
        YGtranslationTotalmotion_sov = hf[f'{path_sov}/YGtranslationTotalmotion'][3000:]
        ZGtranslationTotalmotion_sov = hf[f'{path_sov}/ZGtranslationTotalmotion'][3000:]
        XLrotationTotalmotion_sov = np.deg2rad(hf[f'{path_sov}/XLrotationTotalmotion'][3000:])
        YLrotationTotalmotion_sov = np.deg2rad(hf[f'{path_sov}/YLrotationTotalmotion'][3000:])
        ZGrotationTotalmotion_sov = np.deg2rad(hf[f'{path_sov}/ZGrotationTotalmotion'][3000:])

        XLvelocity_sov = hf[f'{path_velo_sov}/XLvelocityTotalmotion'][3000:]
        YLvelocity_sov = hf[f'{path_velo_sov}/YLvelocityTotalmotion'][3000:]
        ZLvelocity_sov = hf[f'{path_velo_sov}/ZLvelocityTotalmotion'][3000:]
        XLang_vel_sov = np.deg2rad(hf[f'{path_velo_sov}/XLang_vel_Totalmotion'][3000:])
        YLang_vel_sov = np.deg2rad(hf[f'{path_velo_sov}/YLang_vel_Totalmotion'][3000:])
        ZLang_vel_sov = np.deg2rad(hf[f'{path_velo_sov}/ZLang_vel_Totalmotion'][3000:])

        wave_SOV = hf[f'{path_wave}/SOV/Wave elevation/Totalwaveelevation'][3000:]

        global_sov_data = pd.DataFrame({
            'Time': time_array[:len(XGtranslationTotalmotion_sov)],
            'SOV - X Translation': XGtranslationTotalmotion_sov,
            'SOV - Y Translation': YGtranslationTotalmotion_sov,
            'SOV - Z Translation': ZGtranslationTotalmotion_sov,
            'SOV - X Rotation': XLrotationTotalmotion_sov,
            'SOV - Y Rotation': YLrotationTotalmotion_sov,
            'SOV - Z Rotation': ZGrotationTotalmotion_sov,
            'SOV - X Velocity': XLvelocity_sov,
            'SOV - Y Velocity': YLvelocity_sov,
            'SOV - Z Velocity': ZLvelocity_sov,
            'SOV - X Angular Velocity': XLang_vel_sov,
            'SOV - Y Angular Velocity': YLang_vel_sov,
            'SOV - Z Angular Velocity': ZLang_vel_sov
        })

        data['SOV Data'] = global_sov_data

        global_floater_data = pd.DataFrame({
            'Time': time_array[:len(XGtranslationTotalmotion_sov)],
            'floater - X Translation': XGtranslationTotalmotion_floater,
            'floater - Y Translation': YGtranslationTotalmotion_floater,
            'floater - Z Translation': ZGtranslationTotalmotion_floater,
            'floater - X Rotation': XLrotationTotalmotion_floater,
            'floater - Y Rotation': YLrotationTotalmotion_floater,
            'floater - Z Rotation': ZGrotationTotalmotion_floater,
            'floater - X Velocity': XLvelocity_floater,
            'floater - Y Velocity': YLvelocity_floater,
            'floater - Z Velocity': ZLvelocity_floater,
            'floater - X Angular Velocity': XLang_vel_floater,
            'floater - Y Angular Velocity': YLang_vel_floater,
            'floater - Z Angular Velocity': ZLang_vel_floater
        })

        data['floater Data'] = global_floater_data

    except KeyError:
        print(f"Warning: Missing SOV or Floater data for {condition}")
        continue

    try:
        gangway_position = np.array([[0], [0], [15]])
        hookup_position = np.array([[25.875], [-44.817], [15]])
        updated_gangway_positions = []
        updated_hookup_positions = []

        for i in range(len(data['SOV Data'])):
            roll_sov = data['SOV Data']['SOV - X Rotation'].iloc[i]
            pitch_sov = data['SOV Data']['SOV - Y Rotation'].iloc[i]
            yaw_sov = data['SOV Data']['SOV - Z Rotation'].iloc[i]
            R = euler_to_rotation_matrix(roll_sov, pitch_sov, yaw_sov)

        for i in range(len(data['floater Data'])):
            roll_floater = data['floater Data']['floater - X Rotation'].iloc[i]
            pitch_floater = data['floater Data']['floater - Y Rotation'].iloc[i]
            yaw_floater = data['floater Data']['floater - Z Rotation'].iloc[i]
            R = euler_to_rotation_matrix(roll_floater, pitch_floater, yaw_floater)
            updated_gangway_positions.append((R @ gangway_position).flatten())
            updated_hookup_positions.append((R @ hookup_position).flatten())

        updated_gangway_positions_df = pd.DataFrame(updated_gangway_positions, columns=['X_gangway', 'Y_gangway', 'Z_gangway'])
        updated_hookup_positions_df = pd.DataFrame(updated_hookup_positions, columns=['X_hookup', 'Y_hookup', 'Z_hookup'])

        global_hookup_data = pd.DataFrame({
            'X': global_floater_data['floater - X Translation'] + updated_hookup_positions_df['X_hookup'],
            'Y': global_floater_data['floater - Y Translation'] + updated_hookup_positions_df['Y_hookup'],
            'Z': global_floater_data['floater - Z Translation'] + updated_hookup_positions_df['Z_hookup']
        })

        global_gangway_data = pd.DataFrame({
            'X': global_sov_data['SOV - X Translation'] + updated_gangway_positions_df['X_gangway'],
            'Y': global_sov_data['SOV - Y Translation'] + updated_gangway_positions_df['Y_gangway'],
            'Z': global_sov_data['SOV - Z Translation'] + updated_gangway_positions_df['Z_gangway']
        })

        distances = np.sqrt((global_gangway_data['X'] - global_hookup_data['X'])**2 +
                       (global_gangway_data['Y'] - global_hookup_data['Y'])**2 +
                       (global_gangway_data['Z'] - global_hookup_data['Z'])**2)
        
        telescoping = distances - distances.iloc[0]

        luffing = np.degrees(np.arcsin((global_hookup_data['Z']-global_gangway_data['Z'])/distances))
        
        slewing = np.degrees(np.arctan2(global_hookup_data['X'] - global_gangway_data['X'],
                              global_hookup_data['Y'] - global_gangway_data['Y']))
        
        time_segment = [current_time_offset + i * time_step for i in range(len(global_sov_data))]

        # ======================================== VELOCITY =======================================================
        transformed_gangway_velocities = []
        for i in range(len(global_sov_data)):
            v_A = np.array([global_sov_data.loc[i, 'SOV - X Velocity'],
                            global_sov_data.loc[i, 'SOV - Y Velocity'],
                            global_sov_data.loc[i, 'SOV - Z Velocity']])
            omega = np.array([global_sov_data.loc[i, 'SOV - X Angular Velocity'],
                            global_sov_data.loc[i, 'SOV - Y Angular Velocity'],
                            global_sov_data.loc[i, 'SOV - Z Angular Velocity']])
            v_trans = transform_velocity(v_A, omega, gangway_position)
            transformed_gangway_velocities.append(v_trans)
        V_gangway_local = np.stack(transformed_gangway_velocities, axis=0)
        
        theta_x = global_sov_data['SOV - X Rotation'].to_numpy()
        theta_y = global_sov_data['SOV - Y Rotation'].to_numpy()
        theta_z = global_sov_data['SOV - Z Rotation'].to_numpy()
        rotation_matrices = np.array([euler_to_rotation_matrix(rx, ry, rz) 
                                    for rx, ry, rz in zip(theta_x, theta_y, theta_z)])
        V_gangway_global = np.einsum('ijk,ik->ij', rotation_matrices, V_gangway_local)
        
        transformed_hookup_velocities = []
        for i in range(len(global_floater_data)):
            v_A = np.array([global_floater_data.loc[i, 'floater - X Velocity'],
                            global_floater_data.loc[i, 'floater - Y Velocity'],
                            global_floater_data.loc[i, 'floater - Z Velocity']])
            omega = np.array([global_floater_data.loc[i, 'floater - X Angular Velocity'],
                            global_floater_data.loc[i, 'floater - Y Angular Velocity'],
                            global_floater_data.loc[i, 'floater - Z Angular Velocity']])
            v_trans = transform_velocity(v_A, omega, hookup_position)
            transformed_hookup_velocities.append(v_trans)
        V_hookup_global = np.stack(transformed_hookup_velocities, axis=0)


        d_vector_x = global_gangway_data['X'] - global_hookup_data['X']
        d_vector_y = global_gangway_data['Y'] - global_hookup_data['Y']
        d_vector_z = global_gangway_data['Z'] - global_hookup_data['Z']
        distances = np.sqrt(d_vector_x**2 + d_vector_y**2 + d_vector_z**2)
        telescoping_disp = distances

        V_rel = V_gangway_global - V_hookup_global
        unit_dx = d_vector_x / distances
        unit_dy = d_vector_y / distances
        unit_dz = d_vector_z / distances
        telescoping_vel = V_rel[:, 0]*unit_dx + V_rel[:, 1]*unit_dy + V_rel[:, 2]*unit_dz

        luffing_vel = np.gradient(luffing, time_segment)
        slewing_vel = np.gradient(slewing, time_segment)

        gangway_metrics_df = pd.DataFrame({
            'Time': time_segment,
            'wave': wave_Floater[:len(time_segment)],
            'telescoping': telescoping,
            'luffing': luffing,
            'slewing': slewing,
            'telescoping_vel': telescoping_vel,
            'luffing_vel': luffing_vel,
            'slewing_vel': slewing_vel
        })

        if 'combined_gangway_metrics' not in locals():
            combined_gangway_metrics = gangway_metrics_df.copy()
        else:
            combined_gangway_metrics = pd.concat([combined_gangway_metrics, gangway_metrics_df], ignore_index=True)

        XLrotationTotalmotion_floater = np.rad2deg(XLrotationTotalmotion_floater)
        YLrotationTotalmotion_floater = np.rad2deg(YLrotationTotalmotion_floater)
        ZGrotationTotalmotion_floater = np.rad2deg(ZGrotationTotalmotion_floater)
        XLrotationTotalmotion_sov = np.rad2deg(XLrotationTotalmotion_sov)
        YLrotationTotalmotion_sov = np.rad2deg(YLrotationTotalmotion_sov)
        ZGrotationTotalmotion_sov = np.rad2deg(ZGrotationTotalmotion_sov)

        global_gangway_hookup_data = pd.DataFrame({
            'wave_Floater': wave_Floater,
            'wave_SOV': wave_SOV,
            'PtfmTDX_Floater': global_floater_data['floater - X Translation'] + updated_hookup_positions_df['X_hookup'],
            'PtfmTDY_Floater': global_floater_data['floater - Y Translation'] + updated_hookup_positions_df['Y_hookup'],
            'PtfmTDZ_Floater': global_floater_data['floater - Z Translation'] + updated_hookup_positions_df['Z_hookup'],
            'PtfmRDX_Floater': XLrotationTotalmotion_floater,
            'PtfmRDY_Floater': YLrotationTotalmotion_floater,
            'PtfmRDZ_Floater': ZGrotationTotalmotion_floater,
            'PtfmTDX_SOV': global_sov_data['SOV - X Translation'] + updated_gangway_positions_df['X_gangway'],
            'PtfmTDY_SOV': global_sov_data['SOV - Y Translation'] + updated_gangway_positions_df['Y_gangway'],
            'PtfmTDZ_SOV': global_sov_data['SOV - Z Translation'] + updated_gangway_positions_df['Z_gangway'],
            'PtfmRDX_SOV': XLrotationTotalmotion_sov,
            'PtfmRDY_SOV': YLrotationTotalmotion_sov,
            'PtfmRDZ_SOV': ZGrotationTotalmotion_sov
        })

        global_gangway_hookup_data['Time'] = time_segment
        current_time_offset += len(global_sov_data) * time_step
        
        floater_vel_df = pd.DataFrame(V_hookup_global, columns=[
            "Vx_Floater_local", "Vy_Floater_local", "Vz_Floater_local"
        ])
        sov_vel_df = pd.DataFrame(V_gangway_global, columns=[
            "Vx_SOV_local", "Vy_SOV_local", "Vz_SOV_local"
        ])

        floater_ang_vel_df = pd.DataFrame({
            "Wx_Floater": np.rad2deg(global_floater_data['floater - X Angular Velocity']),
            "Wy_Floater": np.rad2deg(global_floater_data['floater - Y Angular Velocity']),
            "Wz_Floater": np.rad2deg(global_floater_data['floater - Z Angular Velocity']),
        })

        sov_ang_vel_df = pd.DataFrame({
            "Wx_SOV": np.rad2deg(global_sov_data['SOV - X Angular Velocity']),
            "Wy_SOV": np.rad2deg(global_sov_data['SOV - Y Angular Velocity']),
            "Wz_SOV": np.rad2deg(global_sov_data['SOV - Z Angular Velocity']),
        })

        combined_velocities = pd.concat([
            floater_vel_df.reset_index(drop=True),
            sov_vel_df.reset_index(drop=True),
            floater_ang_vel_df.reset_index(drop=True),
            sov_ang_vel_df.reset_index(drop=True)
        ], axis=1)

        global_gangway_hookup_data = pd.concat([
            global_gangway_hookup_data.reset_index(drop=True),
            combined_velocities.reset_index(drop=True)
        ], axis=1)

        combined_gangway_data = pd.concat([combined_gangway_data, global_gangway_hookup_data], ignore_index=True)

        d_vector_x = global_gangway_data['X'] - global_hookup_data['X']
        d_vector_y = global_gangway_data['Y'] - global_hookup_data['Y']
        d_vector_z = global_gangway_data['Z'] - global_hookup_data['Z']
        distances = np.sqrt(d_vector_x**2 + d_vector_y**2 + d_vector_z**2)
        telescoping_disp = distances

        V_rel = V_gangway_global - V_hookup_global
        unit_dx = d_vector_x / distances
        unit_dy = d_vector_y / distances
        unit_dz = d_vector_z / distances
        telescoping_vel = V_rel[:, 0]*unit_dx + V_rel[:, 1]*unit_dy + V_rel[:, 2]*unit_dz

        luffing_vel = np.gradient(luffing, time_segment)
        slewing_vel = np.gradient(slewing, time_segment) 


    except KeyError as e:
        print(f"[KeyError] Missing key {e} in condition {condition}")

if combined_gangway_data.empty:
    print("[ERROR] No valid ConditionSets were processed. Check for missing keys or invalid paths.")
    exit(1)

cols = ['Time'] + [col for col in combined_gangway_data.columns if col != 'Time']
combined_gangway_data = combined_gangway_data[cols]

floater_df = combined_gangway_data[[
    'Time', 'wave_Floater',
    'PtfmTDX_Floater', 'PtfmTDY_Floater', 'PtfmTDZ_Floater',
    'PtfmRDX_Floater', 'PtfmRDY_Floater', 'PtfmRDZ_Floater',
    'Vx_Floater_local', 'Vy_Floater_local', 'Vz_Floater_local',
    'Wx_Floater', 'Wy_Floater', 'Wz_Floater'
]].rename(columns={'wave_Floater': 'wave'})

sov_df = combined_gangway_data[[
    'Time', 'wave_SOV',
    'PtfmTDX_SOV', 'PtfmTDY_SOV', 'PtfmTDZ_SOV',
    'PtfmRDX_SOV', 'PtfmRDY_SOV', 'PtfmRDZ_SOV',
    'Vx_SOV_local', 'Vy_SOV_local', 'Vz_SOV_local',
    'Wx_SOV', 'Wy_SOV', 'Wz_SOV'
]].rename(columns={'wave_SOV': 'wave'})

base_path = os.path.splitext(output_csv_path)[0]
floater_path = base_path + "_Option2_Floater.csv"
sov_path = base_path + "_Option2_SOV.csv"
floater_df.to_csv(floater_path, index=False)
sov_df.to_csv(sov_path, index=False)

print(f"Floater CSV saved at: {floater_path}")
print(f"SOV CSV saved at: {sov_path}")

gangway_motion_path = base_path + "_Option3.csv"
combined_gangway_metrics.to_csv(gangway_motion_path, index=False)
print(f"option3 motion CSV saved at: {gangway_motion_path}")

sov_df_no_wave = sov_df.drop(columns=["wave"])
combined_Option1_df = pd.merge(floater_df, sov_df_no_wave, on="Time", how="inner")

Option1_path = base_path + "_Option1.csv"
combined_Option1_df.to_csv(Option1_path, index=False)
print(f"Option1 motion CSV saved at: {Option1_path}")
