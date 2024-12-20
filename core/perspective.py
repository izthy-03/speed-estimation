import numpy as np
import cv2 as cv

from utils.geometry import *


def get_earth_perspective_matrix(
    img_width_px,
    img_height_px,
    focal_length_mm,
    sensor_width_mm,
    sensor_height_mm,
    cam_yaw_deg,
    cam_pitch_deg,
    cam_roll_deg,
):
    """
    Get the perspective matrix for the earth with given camera parameters

    Args:
        img_width_px (int): The width of the image in pixels
        img_height_px (int): The height of the image in pixels
        focal_length_mm (float): The focal length of the camera in mm
        sensor_width_mm (float): The width of the camera sensor in mm
        sensor_height_mm (float): The height of the camera sensor in mm
        cam_yaw_deg (float): The yaw of the camera in degrees, reserved
        cam_pitch_deg (float): The pitch of the camera in degrees
        cam_roll_deg (float): The roll of the camera in degrees, reserved
    
    Returns:
        np.array: The perspective matrix for the earth
    """
    # Calculate the vanishing point of horizon
    vp_x_mm = 0
    vp_y_mm = -focal_length_mm * np.tan(np.radians(cam_pitch_deg))
    if abs(vp_y_mm) <= sensor_height_mm / 2:
        # The vanishing point is within the sensor canvas
        # TODO
        return None

    if vp_y_mm > 0:
        # Pitch < 0, rising point. The vanishing line is above the sensor canvas
        # Right-bottom corner of the sensor canvas
        rb_x_mm = sensor_width_mm / 2
        rb_y_mm = -sensor_height_mm / 2

        A1, B1, C1 = line_equation(vp_x_mm, vp_y_mm, rb_x_mm, rb_y_mm)
        # Upper canvas line: y - sensor_height_mm / 2 = 0
        A2, B2, C2 = 0, 1, -sensor_height_mm / 2

        # Intersection point
        xi1_mm, yi1_mm = line_intersection(A1, B1, C1, A2, B2, C2)
        xi2_mm, yi2_mm = -xi1_mm, yi1_mm

        # Scale to pixels
        xi1_px = (xi1_mm * img_width_px / sensor_width_mm) + img_width_px / 2   
        xi2_px = (xi2_mm * img_width_px / sensor_width_mm) + img_width_px / 2
        yi1_px, yi2_px = img_height_px, img_height_px

        # Perspective matrix
        src = np.array([[0, 0],
                        [img_width_px, 0],
                        [img_width_px, img_height_px],
                        [0, img_height_px]], dtype=np.float32)
        dst = np.array([[0, 0],
                        [img_width_px, 0],
                        [xi2_px, yi2_px],
                        [xi1_px, yi1_px]], dtype=np.float32)

        return cv.getPerspectiveTransform(src, dst)

    return None


def get_length_per_pixel_xy(
    img_width_px,
    img_height_px,
    focal_length_mm,
    sensor_width_mm,
    sensor_height_mm,
    cam_yaw_deg,
    cam_pitch_deg,
    cam_roll_deg,
    cam_height_m,
):
    """
    Get the length(mm) per pixel in x and y direction in mm

    Args:
        img_width_px (int): The width of the image in pixels
        img_height_px (int): The height of the image in pixels
        focal_length_mm (float): The focal length of the camera in mm
        sensor_width_mm (float): The width of the camera sensor in mm
        sensor_height_mm (float): The height of the camera sensor in mm
        cam_yaw_deg (float): The yaw of the camera in degrees, reserved
        cam_pitch_deg (float): The pitch of the camera in degrees
        cam_roll_deg (float): The roll of the camera in degrees, reserved
        cam_height_m (float): The height of the camera in meters
    
    Returns:
        float, float: The length per pixel in x and y direction in mm
    """
    return 11.54, 26.93