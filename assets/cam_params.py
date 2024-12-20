# Camera sensor size in inches. 1 inch = 16mm
CMOS_diagnal_inch = 1 / 1.7

# Full frame(4:3) sensor size in mm
CMOS_full_width_mm = CMOS_diagnal_inch * 16 * 4 / 5
CMOS_full_height_mm = CMOS_diagnal_inch * 16 * 3 / 5


def cmos_size_ratio(width_ratio, height_ratio):
    """
    Get the sensor size in mm given the ratio of the width and height

    Args:
        width_ratio (float): The ratio of the width, e.g. 16
        height_ratio (float): The ratio of the height, e.g. 9

    Returns:
        float: The width of the sensor in mm
        float: The height of the sensor in mm
    """
    if height_ratio/width_ratio > 3/4:
        return CMOS_full_height_mm * width_ratio / height_ratio, CMOS_full_height_mm
    else:
        return CMOS_full_width_mm, CMOS_full_width_mm * height_ratio / width_ratio
    
print(cmos_size_ratio(16, 9))
