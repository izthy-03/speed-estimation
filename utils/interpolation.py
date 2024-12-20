import pandas as pd
from tqdm import tqdm

from collections import deque

def bytetrack_interpolation(
    bytetrack_df: pd.DataFrame,
    window_size: int = 5,
):
    # Sort by tid
    bytetrack_df = bytetrack_df.sort_values(by=["tid", "frame"])

    bt_list = bytetrack_df.to_dict(orient="records")
    bt_len = len(bt_list)

    interplt_list = []

    frame_num = int(bt_list[-1]["frame"]) + 1

    for btid in tqdm(range(1, bt_len)):
        if bt_list[btid]["tid"] != bt_list[btid - 1]["tid"]:
            continue

        interval = bt_list[btid]["frame"] - bt_list[btid - 1]["frame"]
        if interval <= 1 or interval > window_size:
            continue

        # Interpolate the missing frames
        for i in range(1, interval):
            ratio = i / interval
            interplt = {}
            for key in bt_list[btid - 1]:
                if key == "frame":
                    interplt[key] = bt_list[btid - 1][key] + i
                elif key in ["tl_x", "tl_y", "br_x", "br_y"]:
                    interplt[key] = bt_list[btid - 1][key] * (1 - ratio) + bt_list[btid][key] * ratio
                else:
                    interplt[key] = bt_list[btid - 1][key]
        
            interplt_list.append(interplt)

    # Concatenate the interpolated list
    bt_list = bt_list + interplt_list
    # Sort by frame
    bt_list = sorted(bt_list, key=lambda x: x["frame"])
    new_bytetrack_df = pd.DataFrame(bt_list, columns=bytetrack_df.columns)
    
    return new_bytetrack_df



