"""Test code for UQAM gait lab."""

import kineticstoolkit.lab as ktk
import numpy as np
import humans

# %% File selection

static_file = "sample_static.c3d"
gait_file = "sample_gait.c3d"

marker_rename = {
    "RFEP": "RHJC",
    "LFEP": "LHJC",
    "RFEO": "RKJC",
    "LFEO": "LKJC",
    "RTIO": "RAJC",
    "LTIO": "LAJC",
    "*114": "LKneeMedial",
    "*115": "RKneeMedial",
    "*113": "LMalleolusMedial",
    "*116": "RMalleolusMedial",
}

# %% Create joint centers based on a static acquisition

# Load the static acquisition
points = ktk.read_c3d(
    static_file,
    convert_point_unit=True,
)["Points"]

# Rename markers
for marker in marker_rename:
    points.rename_data(marker, marker_rename[marker], in_place=True)

# Fake anthropometric measurements
marker_radius = 0.01
d_knee = (
    np.mean(
        np.sqrt(
            np.sum(
                (points.data["LKNE"] - points.data["LKneeMedial"]) ** 2, axis=1
            )
        )
    )
    - 2 * marker_radius
)
d_ankle = (
    np.mean(
        np.sqrt(
            np.sum(
                (points.data["LANK"] - points.data["LMalleolusMedial"]) ** 2, axis=1
            )
        )
    )
    - 2 * marker_radius
)
l_leg = np.max(
    np.sqrt(np.sum((points.data["RASI"] - points.data["RANK"]) ** 2, axis=1))
)

# Interconnections for visualization
interconnections_cgm = {
    "LowerLimbMarkers": {
        "Links": [
            ["*TOE", "*ANK", "*KNE", "*ASI", "*PSI", "*KNE"],
            ["*TOE", "*HEE", "*ANK"],
        ],
        "Color": [0.25, 0.25, 0.25],
    },
    "PelvisMarkers": {
        "Links": [["RASI", "LASI"], ["RPSI", "LPSI"]],
        "Color": [0.25, 0.25, 0.25],
    },
    "SideMarkers": {
        "Links": [["*ANK", "*TIB"], ["*KNE", "*THI"]],
        "Color": "b",
    },
    "LowerLimbReference": {
        "Links": [["*HJC", "*KJC", "*AJC"]],
        "Color": "y",
    },
    "LowerLimbTest": {
        "Links": [["*HJCTest", "*KJCTest", "*AJCTest"]],
        "Color": "r",
    },
}


points.data["PelvisTest"] = humans.create_pelvis_lcs_davis1991(
    rasis=points.data["RASI"],
    lasis=points.data["LASI"],
    rpsis=points.data["RPSI"],
    lpsis=points.data["LPSI"],
)

# Add hips
for side in ["R", "L"]:
    points.data[f"{side}HJCTest"] = humans.infer_hip_joint_center_hara2016(
        rasis=points.data["RASI"],
        lasis=points.data["LASI"],
        rpsis=points.data["RPSI"],
        lpsis=points.data["LPSI"],
        l_leg=l_leg,
        side=side,
    )
    points.add_data_info(f"{side}HJCTest", "Color", "r", in_place=True)

# Thigh LCS
for side in ["R", "L"]:
    points.data[f"{side}ThighTest"] = humans.create_thigh_lcs_davis1991(
        hjc=points.data[f"{side}HJCTest"],
        lateral_ep=points.data[f"{side}KNE"],
        thigh_marker=points.data[f"{side}THI"],
        side=side,
    )

# Knee joint center
for side in ["R", "L"]:
    points.data[f"{side}KJCTest"] = humans.infer_knee_joint_center_davis1991(
        hjc=points.data[f"{side}HJCTest"],
        lateral_ep=points.data[f"{side}KNE"],
        thigh_marker=points.data[f"{side}THI"],
        knee_width=d_knee,
        marker_radius=marker_radius,
    )
    points.add_data_info(f"{side}KJCTest", "Color", "r", in_place=True)

# Shank LCS
for side in ["R", "L"]:
    points.data[f"{side}ShankTest"] = humans.create_shank_lcs_davis1991(
        kjc=points.data[f"{side}KJCTest"],
        lateral_mal=points.data[f"{side}ANK"],
        shank_marker=points.data[f"{side}TIB"],
        side=side,
    )

# Ankle joint center
for side in ["R", "L"]:
    points.data[f"{side}AJCTest"] = humans.infer_ankle_joint_center_davis1991(
        kjc=points.data[f"{side}KJCTest"],
        lateral_mal=points.data[f"{side}ANK"],
        shank_marker=points.data[f"{side}TIB"],
        ankle_width=d_ankle,
        marker_radius=marker_radius,
    )
    points.add_data_info(f"{side}AJCTest", "Color", "r", in_place=True)


ktk.Player(points, interconnections=interconnections_cgm, up="z", anterior="y")
