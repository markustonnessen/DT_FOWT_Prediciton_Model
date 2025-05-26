import h5py
import pandas as pd
import numpy as np
import os
import math

def extract_and_save_csv(h5_input_arg, timestep):
    if isinstance(h5_input_arg, list):
        if len(h5_input_arg) < 2:
            raise ValueError("Expected at least 1 argument: input.h5")
        input_h5_path = h5_input_arg[1]
    else:
        input_h5_path = h5_input_arg

    if not os.path.exists(input_h5_path):
        raise FileNotFoundError(f"Could not find input file: {input_h5_path}")

    print(f"[INFO] Reading HDF5 file: {input_h5_path}")
    hf = h5py.File(input_h5_path, 'r')

    # === PATHS ===
    path_floater = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/Floater/Global total position'
    path_sov = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/SOV/Global total position'
    path_wave_Floater = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/Floater/Wave elevation'
    path_wave_SOV = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/SOV/Wave elevation'
    path_velo_floater = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/Floater/Local velocity'
    path_velo_sov = 'IEA15MW_VolturnUS_S1/Initial1/Dynamic/SOV/Local velocity'

    # === RAW DATA ===
    wave_Floater = hf[f'{path_wave_Floater}/Totalwaveelevation'][3000:]
    wave_SOV = hf[f'{path_wave_SOV}/Totalwaveelevation'][3000:]

    PtfmTDX_Floater = hf[f'{path_floater}/XGtranslationTotalmotion'][3000:]
    PtfmTDY_Floater = hf[f'{path_floater}/YGtranslationTotalmotion'][3000:]
    PtfmTDZ_Floater = hf[f'{path_floater}/ZGtranslationTotalmotion'][3000:]
    PtfmRDX_Floater = np.deg2rad(hf[f'{path_floater}/XLrotationTotalmotion'][3000:])
    PtfmRDY_Floater = np.deg2rad(hf[f'{path_floater}/YLrotationTotalmotion'][3000:])
    PtfmRDZ_Floater = np.deg2rad(hf[f'{path_floater}/ZGrotationTotalmotion'][3000:])

    PtfmTDX_SOV = hf[f'{path_sov}/XGtranslationTotalmotion'][3000:]
    PtfmTDY_SOV = hf[f'{path_sov}/YGtranslationTotalmotion'][3000:]
    PtfmTDZ_SOV = hf[f'{path_sov}/ZGtranslationTotalmotion'][3000:]
    PtfmRDX_SOV = np.deg2rad(hf[f'{path_sov}/XLrotationTotalmotion'][3000:])
    PtfmRDY_SOV = np.deg2rad(hf[f'{path_sov}/YLrotationTotalmotion'][3000:])
    PtfmRDZ_SOV = np.deg2rad(hf[f'{path_sov}/ZGrotationTotalmotion'][3000:])

    XLvelocity_SOV = hf[f'{path_velo_sov}/XLvelocityTotalmotion'][3000:]
    YLvelocity_SOV = hf[f'{path_velo_sov}/YLvelocityTotalmotion'][3000:]
    ZLvelocity_SOV = hf[f'{path_velo_sov}/ZLvelocityTotalmotion'][3000:]
    XLang_vel_SOV = np.deg2rad(hf[f'{path_velo_sov}/XLang_vel_Totalmotion'][3000:])
    YLang_vel_SOV = np.deg2rad(hf[f'{path_velo_sov}/YLang_vel_Totalmotion'][3000:])
    ZLang_vel_SOV = np.deg2rad(hf[f'{path_velo_sov}/ZLang_vel_Totalmotion'][3000:])

    XLvelocity_Floater = hf[f'{path_velo_floater}/XLvelocityTotalmotion'][3000:]
    YLvelocity_Floater = hf[f'{path_velo_floater}/YLvelocityTotalmotion'][3000:]
    ZLvelocity_Floater = hf[f'{path_velo_floater}/ZLvelocityTotalmotion'][3000:]
    XLang_vel_Floater = np.deg2rad(hf[f'{path_velo_floater}/XLang_vel_Totalmotion'][3000:])
    YLang_vel_Floater = np.deg2rad(hf[f'{path_velo_floater}/YLang_vel_Totalmotion'][3000:])
    ZLang_vel_Floater = np.deg2rad(hf[f'{path_velo_floater}/ZLang_vel_Totalmotion'][3000:])

    num_steps = len(wave_Floater)
    time_array = [i * timestep for i in range(num_steps)]
    sim_length = time_array[-1]

    # === ROTATION MATRIX ===
    def euler_to_rotation_matrix(roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    

    def transform_velocity(v_A, omega, r_BA):
        omega_cross_r = np.cross(omega, r_BA.flatten())
        return v_A + omega_cross_r


    # === TRANSFORMATIONS ===
    gangway_position = np.array([[0], [0], [15]])
    hookup_position = np.array([[25.875], [-44.817], [15]])

    updated_gangway_positions = []
    updated_hookup_positions = []

    for i in range(num_steps):
        R_sov = euler_to_rotation_matrix(PtfmRDX_SOV[i], PtfmRDY_SOV[i], PtfmRDZ_SOV[i])
        R_floater = euler_to_rotation_matrix(PtfmRDX_Floater[i], PtfmRDY_Floater[i], PtfmRDZ_Floater[i])

        gangway_transformed = R_sov @ gangway_position
        hookup_transformed = R_floater @ hookup_position

        updated_gangway_positions.append(gangway_transformed.flatten())
        updated_hookup_positions.append(hookup_transformed.flatten())

    updated_gangway_df = pd.DataFrame(updated_gangway_positions, columns=['X_gangway', 'Y_gangway', 'Z_gangway'])
    updated_hookup_df = pd.DataFrame(updated_hookup_positions, columns=['X_hookup', 'Y_hookup', 'Z_hookup'])

    PtfmRDX_Floater = np.rad2deg(PtfmRDX_Floater)
    PtfmRDY_Floater = np.rad2deg(PtfmRDY_Floater)
    PtfmRDZ_Floater = np.rad2deg(PtfmRDZ_Floater)

    PtfmRDX_SOV = np.rad2deg(PtfmRDX_SOV)
    PtfmRDY_SOV = np.rad2deg(PtfmRDY_SOV)
    PtfmRDZ_SOV = np.rad2deg(PtfmRDZ_SOV)

    df_wave_Floater = pd.DataFrame({
        'Time': time_array,
        'wave': wave_Floater
    })

    df_wave_SOV = pd.DataFrame({
        'Time': time_array,
        'wave': wave_SOV
    })

    df_motion_Floater = pd.DataFrame({
        'Time': time_array,
        'PtfmTDX_Floater': PtfmTDX_Floater + updated_hookup_df['X_hookup'],
        'PtfmTDY_Floater': PtfmTDY_Floater + updated_hookup_df['Y_hookup'],
        'PtfmTDZ_Floater': PtfmTDZ_Floater + updated_hookup_df['Z_hookup'],
        'PtfmRDX_Floater': PtfmRDX_Floater,
        'PtfmRDY_Floater': PtfmRDY_Floater,
        'PtfmRDZ_Floater': PtfmRDZ_Floater
    })


    df_motion_SOV = pd.DataFrame({
        'Time': time_array,
        'PtfmTDX_SOV': PtfmTDX_SOV + updated_gangway_df['X_gangway'],
        'PtfmTDY_SOV': PtfmTDY_SOV + updated_gangway_df['Y_gangway'],
        'PtfmTDZ_SOV': PtfmTDZ_SOV + updated_gangway_df['Z_gangway'],
        'PtfmRDX_SOV': PtfmRDX_SOV,
        'PtfmRDY_SOV': PtfmRDY_SOV,
        'PtfmRDZ_SOV': PtfmRDZ_SOV
    })

    script_dir = os.path.abspath(os.path.dirname(__file__))
    incoming_path = os.path.join(script_dir, "Incoming_Waves")
    os.makedirs(incoming_path, exist_ok=True)

    wave_file_path_Floater = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_wave_Option2_Floater.csv")
    motion_file_path_Floater = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_motion_Option2_Floater.csv")

    wave_file_path_SOV = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_wave_Option2_SOV.csv")
    motion_file_path_SOV = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_motion_Option2_SOV.csv")

    df_wave_Floater.to_csv(wave_file_path_Floater, index=False)
    df_motion_Floater.to_csv(motion_file_path_Floater, index=False)

    df_wave_SOV.to_csv(wave_file_path_SOV, index=False)
    df_motion_SOV.to_csv(motion_file_path_SOV, index=False)

    print(f"[INFO] Saved FLOATER wave CSV to {wave_file_path_Floater}")
    print(f"[INFO] Saved FLOATER motion CSV to {motion_file_path_Floater}")
    print(f"[INFO] Saved SOV wave CSV to {wave_file_path_SOV}")
    print(f"[INFO] Saved SOV motion CSV to {motion_file_path_SOV}")

    distances = np.sqrt(
        (df_motion_SOV['PtfmTDX_SOV'] - df_motion_Floater['PtfmTDX_Floater'])**2 +
        (df_motion_SOV['PtfmTDY_SOV'] - df_motion_Floater['PtfmTDY_Floater'])**2 +
        (df_motion_SOV['PtfmTDZ_SOV'] - df_motion_Floater['PtfmTDZ_Floater'])**2
    )

    telescoping = distances - distances.iloc[0]
    luffing = np.degrees(np.arcsin((df_motion_Floater['PtfmTDZ_Floater'] - df_motion_SOV['PtfmTDZ_SOV'])/distances))
    delta_x = df_motion_Floater['PtfmTDX_Floater'] - df_motion_SOV['PtfmTDX_SOV']
    delta_y = df_motion_Floater['PtfmTDY_Floater'] - df_motion_SOV['PtfmTDY_SOV']
    slewing_angle = np.degrees(np.arctan2(delta_x, delta_y))
    slewing = slewing_angle
  


    df_motion_Gangway = pd.DataFrame({
        'Time': time_array,
        'telescoping': telescoping,
        'luffing': luffing,
        'slewing': slewing
    })

    V_gangway_global = []
    V_hookup_global = []

    for i in range(num_steps):
        v_A_sov = np.array([XLvelocity_SOV[i], YLvelocity_SOV[i], ZLvelocity_SOV[i]])
        omega_sov = np.array([XLang_vel_SOV[i], YLang_vel_SOV[i], ZLang_vel_SOV[i]])
        v_local = transform_velocity(v_A_sov, omega_sov, gangway_position)
        R_sov = euler_to_rotation_matrix(np.deg2rad(PtfmRDX_SOV[i]), np.deg2rad(PtfmRDY_SOV[i]), np.deg2rad(PtfmRDZ_SOV[i]))
        V_gangway_global.append(R_sov @ v_local)

        v_A_floater = np.array([XLvelocity_Floater[i], YLvelocity_Floater[i], ZLvelocity_Floater[i]])
        omega_floater = np.array([XLang_vel_Floater[i], YLang_vel_Floater[i], ZLang_vel_Floater[i]])
        v_local = transform_velocity(v_A_floater, omega_floater, hookup_position)
        R_floater = euler_to_rotation_matrix(np.deg2rad(PtfmRDX_Floater[i]), np.deg2rad(PtfmRDY_Floater[i]), np.deg2rad(PtfmRDZ_Floater[i]))
        V_hookup_global.append(R_floater @ v_local)

    V_gangway_global = np.array(V_gangway_global)
    V_hookup_global = np.array(V_hookup_global)
    V_rel = V_gangway_global - V_hookup_global

    df_motion_Floater[['Vx_Floater_local', 'Vy_Floater_local', 'Vz_Floater_local']] = V_hookup_global
    df_motion_SOV[['Vx_SOV_local', 'Vy_SOV_local', 'Vz_SOV_local']] = V_gangway_global

    df_motion_Floater[['Wx_Floater', 'Wy_Floater', 'Wz_Floater']] = np.rad2deg(
        np.column_stack([XLang_vel_Floater, YLang_vel_Floater, ZLang_vel_Floater])
    )
    df_motion_SOV[['Wx_SOV', 'Wy_SOV', 'Wz_SOV']] = np.rad2deg(
        np.column_stack([XLang_vel_SOV, YLang_vel_SOV, ZLang_vel_SOV])
    )

    dx = df_motion_SOV['PtfmTDX_SOV'] - df_motion_Floater['PtfmTDX_Floater']
    dy = df_motion_SOV['PtfmTDY_SOV'] - df_motion_Floater['PtfmTDY_Floater']
    dz = df_motion_SOV['PtfmTDZ_SOV'] - df_motion_Floater['PtfmTDZ_Floater']
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    ux, uy, uz = dx / dist, dy / dist, dz / dist

    telescoping_vel = V_rel[:, 0] * ux + V_rel[:, 1] * uy + V_rel[:, 2] * uz
    luffing_vel = np.gradient(luffing, timestep)
    slewing_vel = np.gradient(slewing, timestep)

    df_motion_Gangway["telescoping_vel"] = telescoping_vel
    df_motion_Gangway["luffing_vel"] = luffing_vel
    df_motion_Gangway["slewing_vel"] = slewing_vel


    df_wave_Gangway = pd.DataFrame({
        'Time': time_array,
        'wave': wave_Floater
    })

    wave_file_path_Gangway = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_wave_Option3.csv")
    df_wave_Gangway.to_csv(wave_file_path_Gangway, index=False)

    motion_file_path_Gangway = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_motion_Option3.csv")
    df_motion_Gangway.to_csv(motion_file_path_Gangway, index=False)
    print(f"[INFO] Saved Option3 wave CSV to {wave_file_path_Gangway}")
    print(f"[INFO] Saved Option3 motion CSV to {motion_file_path_Gangway}")

    df_motion_Option1 = pd.concat([
        df_motion_Floater.drop(columns=['Time']),
        df_motion_SOV.drop(columns=['Time'])
    ], axis=1)
    df_motion_Option1.insert(0, 'wave', wave_Floater)
    df_motion_Option1.insert(0, 'Time', time_array)

    motion_file_path_Option1 = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_motion_Option1.csv")
    df_motion_Option1.to_csv(motion_file_path_Option1, index=False)
    print(f"[INFO] Saved Option1 motion CSV to {motion_file_path_Option1}")

    df_motion_Floater.to_csv(motion_file_path_Floater, index=False)
    df_motion_SOV.to_csv(motion_file_path_SOV, index=False)

    wave_file_path_Option1 = os.path.join(incoming_path, f"{math.ceil(sim_length)}s_wave_Option1.csv")
    df_wave_Floater.to_csv(wave_file_path_Option1, index=False)
    print(f"[INFO] Saved Option1 wave CSV to {wave_file_path_Option1}")



    return (
    df_motion_Floater,
    df_motion_SOV,
    wave_file_path_Floater,
    motion_file_path_Floater,
    wave_file_path_SOV,
    motion_file_path_SOV,
    sim_length,
    df_motion_Gangway,
    motion_file_path_Gangway,
    wave_file_path_Gangway,
    df_motion_Option1,
    motion_file_path_Option1,
    wave_file_path_Option1
)