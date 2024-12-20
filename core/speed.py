import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque

from assets.cam_params import *
from core.perspective import *
from utils.srt import SRT_reader


def pixel_speed_with_fixed_perspect(
    bytetrack_df,
    perspective_matrix, 
    averaging_frame=5
):
    """
    Calculate the average speed of the targets in the video.
    The perspective matrix is fixed.
    
    Args:
        bytetrack_df (pd.DataFrame): The ByteTrack result DataFrame
        perspective_matrix (np.ndarray): The fixed perspective matrix
        averaging_frame (int): The number of frames to average the speed
    
    Returns:
        pd.DataFrame: The DataFrame of the speed of the targets(Unit: pixel/frame)
    """
    speed_df = pd.DataFrame(columns=["frame", "tid", "vx_px", "vy_px"])
    speed_list = []

    bt_len = len(bytetrack_df)
    print(bytetrack_df)

    frame_num = int(bytetrack_df.iloc[-1][0]) + 1
    print(f"Total frames: {frame_num}")

    # Fixed frame-len buffer to calculate the speed
    buf = deque()
    tail = 0

    for frame_id in tqdm(range(frame_num)):
        # Append new frames 
        while tail < bt_len and bytetrack_df.iloc[tail]["frame"] < frame_id + averaging_frame:
            buf.append(bytetrack_df.iloc[tail])
            tail += 1
        
        # Drop old frames
        while buf and buf[0]["frame"] < frame_id:
            buf.popleft()

        # Convert deque to list for processing
        buf_list = list(buf)

        # Calculate average speed in frame_id
        frame_targets = [target for target in buf_list if target["frame"] == frame_id]
        for target in frame_targets:
            tid = target["tid"]
            # Find last tid in buf
            last_target = next((item for item in reversed(buf_list) if item["tid"] == tid and item["frame"] != frame_id), None)
            if last_target is None:
                continue

            # Calculate the speed
            cx1, cy1 = (target["tl_x"] + target["br_x"]) / 2, (target["tl_y"] + target["br_y"]) / 2
            cx1_t, cy1_t = cv.perspectiveTransform(np.array([[[cx1, cy1]]], dtype=np.float32), perspective_matrix)[0][0]
            cx2, cy2 = (last_target["tl_x"] + last_target["br_x"]) / 2, (last_target["tl_y"] + last_target["br_y"]) / 2
            cx2_t, cy2_t = cv.perspectiveTransform(np.array([[[cx2, cy2]]], dtype=np.float32), perspective_matrix)[0][0]

            vx_px = (cx2_t - cx1_t) / (last_target["frame"] - frame_id)
            vy_px = (cy2_t - cy1_t) / (last_target["frame"] - frame_id)

            speed_list.append([frame_id, tid, vx_px, vy_px])

    speed_df = pd.DataFrame(speed_list, columns=["frame", "tid", "vx_px", "vy_px"])
    return speed_df


def pixel_speed_to_true_speed(
    speed_df: pd.DataFrame,
    lenx_m_ppx: float,
    leny_m_ppx: float,
    fps: int
):
    """
    Convert the pixel speed to the true speed in m/s.
    
    Args:
        speed_df (pd.DataFrame): The DataFrame of the speed of the targets(Unit: pixel/frame)
        lenx_m_ppx (float): The length of the x-axis in meters per pixel
        leny_m_ppx (float): The length of the y-axis in meters per pixel
        fps (int): The frame rate of the video
    
    Returns:
        pd.DataFrame: The DataFrame of the speed of the targets(Unit: m/s)
    """
    speed_df["vx_m"] = speed_df["vx_px"] * lenx_m_ppx * fps
    speed_df["vy_m"] = speed_df["vy_px"] * leny_m_ppx * fps

    # For frame 6000-end
    for i in range(len(speed_df)):
        if speed_df.loc[i, "frame"] >= 6000:
            speed_df.loc[i, "vy_m"] = speed_df.loc[i, "vy_px"] * 0.07 * fps
    
    return speed_df

def speed_fix_drone(
    speed_df: pd.DataFrame,
    srt: SRT_reader,
):
    """
    Fix the car speed with the drone speed.
    """
    # DataFrame to dict list 
    speed_list = speed_df.to_dict("records")
    speed_list_len = len(speed_list)
    
    frame_num = speed_list[-1]["frame"]
    tail = 0
    for frame_id in range(frame_num):
        drone_speedx = srt.get_float(frame_id+1, "drone_speedx")
        drone_speedy = srt.get_float(frame_id+1, "drone_speedy")

        gb_yaw = srt.get_float(frame_id+1, "gb_yaw")
        if gb_yaw > 0:
            gb_yaw -= 360
        gb_yaw = np.deg2rad(gb_yaw)
        rotation_matrix = np.array([[np.cos(gb_yaw), np.sin(gb_yaw)],
                                    [-np.sin(gb_yaw), np.cos(gb_yaw)]])
        drone_speed_vec = np.array([drone_speedx, drone_speedy])
        drone_speed_vec = np.matmul(rotation_matrix, drone_speed_vec)

        # Fix cars' speed in frame_id
        while tail < speed_list_len and speed_list[tail]["frame"] == frame_id:
            speed_list[tail]["vy_m"] -= drone_speed_vec[0]
            speed_list[tail]["vx_m"] += drone_speed_vec[1]
            tail += 1
    
    # To DataFrame
    speed_df = pd.DataFrame(speed_list, columns=speed_df.columns)

    # Add column kmph
    speed_df["vx_kmph"] = speed_df["vx_m"] * 3.6
    speed_df["vy_kmph"] = speed_df["vy_m"] * 3.6

    # Add abs speed
    speed_df["v_kmph"] = (speed_df["vx_kmph"] ** 2 + speed_df["vy_kmph"] ** 2) ** 0.5 

    return speed_df


def speed_smoonthen(
    speed_df: pd.DataFrame,
    smoothing_frame=5
):
    """
    Smooth the speed with the average speed of the previous frames.
    
    Args:
        speed_df (pd.DataFrame): The DataFrame of the speed of the targets(Unit: m/s)
        smoothing_frame (int): The number of frames to smooth the speed
    
    Returns:
        pd.DataFrame: The DataFrame of the speed of the targets(Unit: m/s)
    """
    # Sort by tid
    speed_df = speed_df.sort_values(by=["tid", "frame"])

    speed_list = speed_df.to_dict("records")
    speed_list_len = len(speed_list)

    buf = deque()
    buf.append(speed_list[0])

    for i in range(1, speed_list_len):
        if speed_list[i]["tid"] != speed_list[i-1]["tid"]:
            buf.clear()
        buf.append(speed_list[i])
        while len(buf) > smoothing_frame:
            buf.popleft()
        
        # Calculate the average speed
        vx_m = sum([item["vx_m"] for item in buf]) / len(buf)
        vy_m = sum([item["vy_m"] for item in buf]) / len(buf)

        speed_list[i]["vx_m"] = vx_m
        speed_list[i]["vy_m"] = vy_m

    # Sort back
    speed_df = pd.DataFrame(speed_list, columns=speed_df.columns)
    speed_df = speed_df.sort_values(by=["frame", "tid"])
    return speed_df
    
