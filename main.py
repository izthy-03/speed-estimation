import cv2 as cv
import pandas as pd

from utils.srt import SRT_reader
from utils.renderer import Renderer
from assets.cam_params import *
from core.perspective import get_earth_perspective_matrix
from core.speed import *


origin_video_path = "../drone/origin.mp4"
srt_file_path = "./data/origin.SRT"
srt = SRT_reader(srt_file_path)


def draw_earth_perspective():
    gimbal_pitch = srt.get_float(0, "gb_pitch")
    focal_len = srt.get_float(0, "focal_len")
    print(f"pitch: {gimbal_pitch}, focal_len: {focal_len}")

    cmos_width, cmos_height = cmos_size_ratio(1920, 1080)
 
    perspective_matrix = get_earth_perspective_matrix(
        img_width_px=1920,
        img_height_px=1080,
        focal_length_mm=focal_len,
        sensor_width_mm=cmos_width,
        sensor_height_mm=cmos_height,
        cam_yaw_deg=0,
        cam_pitch_deg=gimbal_pitch,
        cam_roll_deg=0,
    )

    perspective_matrix = cv.getPerspectiveTransform(
        np.array([[888, 225],
                  [1063, 225],
                  [1210, 1080],
                  [486, 1080]], dtype=np.float32),
        np.array([[888, 225],
                  [1063, 225],
                 [1063, 1080],
                 [888, 1080]], dtype=np.float32)
    )

    print(perspective_matrix)

    def perspective_rectify(frame, frame_id):
        return cv.warpPerspective(frame, perspective_matrix, (1920, 1080))

    renderer = Renderer(origin_video_path)
    renderer.register_painter(perspective_rectify)

    renderer.start("./perspective3.mp4")


def draw_speed():
    # bytetrack_path = "./data/detections.txt"
    # bytetrack_df = pd.read_csv(bytetrack_path, header=None)
    bytetrack_path = "./data/interpolated_detections.txt"
    bytetrack_df = pd.read_csv(bytetrack_path)
    bytetrack_df.columns = ["frame", "tl_x", "tl_y", "br_x", "br_y", "category", "score", "tid"]

    gimbal_pitch = srt.get_float(0, "gb_pitch")
    focal_len = srt.get_float(0, "focal_len")
    print(f"pitch: {gimbal_pitch}, focal_len: {focal_len}")

    cmos_width, cmos_height = cmos_size_ratio(1920, 1080)
 
    # perspective_matrix = get_earth_perspective_matrix(
    #     img_width_px=1920,
    #     img_height_px=1080,
    #     focal_length_mm=focal_len,
    #     sensor_width_mm=cmos_width,
    #     sensor_height_mm=cmos_height,
    #     cam_yaw_deg=0,
    #     cam_pitch_deg=gimbal_pitch,
    #     cam_roll_deg=0,
    # )

    perspective_matrix = cv.getPerspectiveTransform(
        np.array([[785, 440],
                  [1101, 440],
                  [1210, 1080],
                  [486, 1080]], dtype=np.float32),
        np.array([[785, 440],
                  [1101, 440],
                 [1101, 1080],
                 [785, 1080]], dtype=np.float32)
    )

    print(perspective_matrix)

    speed_df = pixel_speed_with_fixed_perspect(bytetrack_df, perspective_matrix)
    # speed_df = pd.read_csv("./data/speed_fin.csv")
    speed_df = pixel_speed_to_true_speed(speed_df, 0.0745, 0.1236, 30)
    speed_df = speed_smoonthen(speed_df)
    speed_df = speed_fix_drone(speed_df, srt)
    speed_df.to_csv("./data/speed_fin.csv", index=False)


    def speed_painter(frame, frame_id):
        bytetrack_slice = bytetrack_df[bytetrack_df["frame"] == frame_id]
        speed_slice = speed_df[speed_df["frame"] == frame_id]

        for _, row in bytetrack_slice.iterrows():
            tid = int(row["tid"])
            # Draw the bounding box
            cv.rectangle(frame, (int(row["tl_x"]), int(row["tl_y"])), (int(row["br_x"]), int(row["br_y"])),
                         (255, 0, 0), 2)

            # Draw the target id
            cv.putText(frame, f"ID: {tid}", (int(row["tl_x"]), int(row["tl_y"])), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

            # Draw speed
            speed = speed_slice[speed_slice["tid"] == row["tid"]]
            if speed.empty:
                continue
            v_kmph = speed["vy_kmph"].values[0]
            color = (0, 204, 0) if abs(v_kmph) > 5 else (0, 0, 255)
            cv.putText(frame, f"{v_kmph:.1f} km/h", (int(row["tl_x"]), int(row["br_y"])), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
            # Draw red bounding box if the speed is less than 5 km/h
            if abs(v_kmph) < 5:
                cv.rectangle(frame, (int(row["tl_x"]), int(row["tl_y"])), (int(row["br_x"]), int(row["br_y"])),
                             (0, 0, 255), 2)

            # Draw pixel speed
            # speed = speed_slice[speed_slice["tid"] == row["tid"]]
            # if speed.empty:
            #     continue
            # vx_px = speed["vx_px"].values[0] * 30
            # vy_px = speed["vy_px"].values[0] * 30
            # color = (0, 204, 0)
            # if tid in [5472, 6529, 6530]:
            #     color = (0, 0, 255)
            # cv.putText(frame, f"vx: {vx_px:.1f} px/s", (int(row["tl_x"]), int(row["br_y"])),
            #               cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
            # cv.putText(frame, f"vy: {vy_px:.1f} px/s", (int(row["tl_x"]), int(row["br_y"] + 20)),
            #                 cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        
        return frame


    def drone_speed_painter(frame ,frame_id):
        drone_speedx = srt.get_float(frame_id+1, "drone_speedx")
        drone_speedy = srt.get_float(frame_id+1, "drone_speedy")

        gb_yaw = srt.get_float(frame_id+1, "gb_yaw")
        if gb_yaw is None:
            return frame
        if gb_yaw > 0:
            gb_yaw -= 360
        gb_yaw = np.deg2rad(gb_yaw)
        rotation_matrix = np.array([[np.cos(gb_yaw), np.sin(gb_yaw)],
                                    [-np.sin(gb_yaw), np.cos(gb_yaw)]])
        drone_speed_vec = np.array([drone_speedx, drone_speedy])
        drone_speed_vec = np.matmul(rotation_matrix, drone_speed_vec)

        color = (0, 0, 255)
        cv.putText(frame, f"drone_speedx_fixed: {drone_speed_vec[0]:.2f} m/s", (1400, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
        cv.putText(frame, f"drone_speedy_fixed: {drone_speed_vec[1]:.2f} m/s", (1400, 80),
                    cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
        return frame

    renderer = Renderer(origin_video_path) 
    renderer.register_painter(speed_painter)
    renderer.register_painter(drone_speed_painter)
    # renderer.start("./speed_correct.mp4")
    # renderer.start("./speed_pixel.mp4")
    renderer.start("./speed_y.mp4")

if __name__ == "__main__":
    # draw_earth_perspective()
    draw_speed()