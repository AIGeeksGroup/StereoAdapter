import cv2
import numpy as np
from pathlib import Path

root_dir = Path("/date2/zhang_h/stereo/tartanair/tartanair_part")

subfolders = [
    item for item in root_dir.iterdir() if item.is_dir() and "soulcity" not in item.name
]

# Initialize global min/max tracking
global_min_depth = float("inf")
global_max_depth = 0.0

for i, subfolder in enumerate(subfolders):
    print(f"Processing subfolder {i + 1}/{len(subfolders)}: {subfolder.name}")

    sample_path = subfolder / subfolder.name / "imgs"
    dpt_lft_dir = sample_path / "depth_02" / "data"
    dpt_rgt_dir = sample_path / "depth_03" / "data"

    # Process left depth images
    for file in dpt_lft_dir.iterdir():
        depth = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

        # Remove zeros and get valid depth values
        valid_depth = depth[depth > 0]

        if len(valid_depth) > 0:
            local_min = np.min(valid_depth)
            local_max = np.max(valid_depth)

            global_min_depth = min(global_min_depth, local_min)
            global_max_depth = max(global_max_depth, local_max)

    # Process right depth images
    for file in dpt_rgt_dir.iterdir():
        depth = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

        # Remove zeros and get valid depth values
        valid_depth = depth[depth > 0]

        if len(valid_depth) > 0:
            local_min = np.min(valid_depth)
            local_max = np.max(valid_depth)

            global_min_depth = min(global_min_depth, local_min)
            global_max_depth = max(global_max_depth, local_max)

print(f"\nGlobal depth statistics (ignoring zeros):")
print(f"Minimum depth: {global_min_depth}")
print(f"Maximum depth: {global_max_depth}")

# Convert to disparity using TartanAir formula
min_disparity = 80.0 / global_max_depth  # furthest object
max_disparity = 80.0 / global_min_depth  # closest object
print(f"\nCorresponding disparity range:")
print(f"Minimum disparity: {min_disparity}")
print(f"Maximum disparity: {max_disparity}")
