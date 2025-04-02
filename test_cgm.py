"""Test code for UQAM gait lab."""

import kineticstoolkit.lab as ktk
import numpy as np
import matplotlib.pyplot as plt
import humans

# %% File selection

static_file = "sample_static.c3d"
gait_file = "sample_gait.c3d"

marker_rename = {
    "RFEP": "RHJC_control",
    "LFEP": "LHJC_control",
    "RFEO": "RKJC_control",
    "LFEO": "LKJC_control",
    "RTIO": "RAJC_control",
    "LTIO": "LAJC_control",
    "*114": "LKNE_med",
    "*115": "RKNE_med",
    "*113": "LANK_med",
    "*116": "RANK_med",
}

# %% Create joint centers based on a static acquisition

# Load the static acquisition
static_points = ktk.read_c3d(
    static_file,
    convert_point_unit=True,
)["Points"]

# Rename markers
for marker in marker_rename:
    static_points.rename_data(marker, marker_rename[marker], in_place=True)

# Fake anthropometric measurements
marker_radius = 0.01
d_knee = (
    np.mean(
        np.sqrt(
            np.sum(
                (static_points.data["LKNE"] - static_points.data["LKNE_med"])
                ** 2,
                axis=1,
            )
        )
    )
    - 2 * marker_radius
)
d_ankle = (
    np.mean(
        np.sqrt(
            np.sum(
                (static_points.data["LANK"] - static_points.data["LANK_med"])
                ** 2,
                axis=1,
            )
        )
    )
    - 2 * marker_radius
)
l_leg = np.max(
    np.sqrt(
        np.sum(
            (static_points.data["RASI"] - static_points.data["RANK"]) ** 2,
            axis=1,
        )
    )
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
        "Links": [["*HJC_control", "*KJC_control", "*AJC_control"]],
        "Color": "y",
    },
    "LowerLimbTest": {
        "Links": [["*HJC", "*KJC", "*AJC"]],
        "Color": "r",
    },
}


static_points.data["Pelvis"] = humans.create_pelvis_lcs_davis1991(
    rasis=static_points.data["RASI"],
    lasis=static_points.data["LASI"],
    rpsis=static_points.data["RPSI"],
    lpsis=static_points.data["LPSI"],
)

# Add hips
for side in ["R", "L"]:
    static_points.data[f"{side}HJC"] = humans.infer_hip_joint_center_hara2016(
        rasis=static_points.data["RASI"],
        lasis=static_points.data["LASI"],
        rpsis=static_points.data["RPSI"],
        lpsis=static_points.data["LPSI"],
        l_leg=l_leg,
        side=side,
    )
    static_points.add_data_info(f"{side}HJC", "Color", "r", in_place=True)

# Thigh LCS
for side in ["R", "L"]:
    static_points.data[f"{side}Thigh"] = humans.create_thigh_lcs_davis1991(
        hjc=static_points.data[f"{side}HJC"],
        lateral_ep=static_points.data[f"{side}KNE"],
        thigh_marker=static_points.data[f"{side}THI"],
        side=side,
    )

# Knee joint center
for side in ["R", "L"]:
    # static_points.data[f"{side}KJC"] = humans.infer_knee_joint_center_davis1991(
    #     hjc=static_points.data[f"{side}HJC"],
    #     lateral_ep=static_points.data[f"{side}KNE"],
    #     thigh_marker=static_points.data[f"{side}THI"],
    #     knee_width=d_knee,
    #     marker_radius=marker_radius,
    # )
    # static_points.add_data_info(f"{side}KJC", "Color", "r", in_place=True)
    static_points.add_data(
        f"{side}KJC",
        0.5
        * (
            static_points.data[f"{side}KNE"]
            + static_points.data[f"{side}KNE_med"]
        ),
        in_place=True,
    )

# Shank LCS
for side in ["R", "L"]:
    static_points.data[f"{side}Shank"] = humans.create_shank_lcs_davis1991(
        kjc=static_points.data[f"{side}KJC"],
        lateral_mal=static_points.data[f"{side}ANK"],
        shank_marker=static_points.data[f"{side}TIB"],
        side=side,
    )

# Ankle joint center
for side in ["R", "L"]:
    # static_points.data[f"{side}AJC"] = humans.infer_ankle_joint_center_davis1991(
    #     kjc=static_points.data[f"{side}KJC"],
    #     lateral_mal=static_points.data[f"{side}ANK"],
    #     shank_marker=static_points.data[f"{side}TIB"],
    #     ankle_width=d_ankle,
    #     marker_radius=marker_radius,
    # )
    # static_points.add_data_info(f"{side}AJC", "Color", "r", in_place=True)
    static_points.add_data(
        f"{side}AJC",
        0.5
        * (
            static_points.data[f"{side}ANK"]
            + static_points.data[f"{side}ANK_med"]
        ),
        in_place=True,
    )


ktk.Player(
    static_points, interconnections=interconnections_cgm, up="z", anterior="y"
)

# %% Create the marker clusters for reconstruction during gait
clusters = {
    "Pelvis": ktk.kinematics.create_cluster(
        static_points, ["RASI", "LASI", "RPSI", "LPSI", "RHJC", "LHJC"]
    ),
    "RThigh": ktk.kinematics.create_cluster(
        static_points, ["RHJC", "RTHI", "RKNE", "RKJC"]
    ),
    "LThigh": ktk.kinematics.create_cluster(
        static_points, ["LHJC", "LTHI", "LKNE", "LKJC"]
    ),
    "RShank": ktk.kinematics.create_cluster(
        static_points, ["RKJC", "RTIB", "RANK", "RAJC"]
    ),
    "LShank": ktk.kinematics.create_cluster(
        static_points, ["LKJC", "LTIB", "LANK", "LAJC"]
    ),
}

# %% Process the gait trial
gait_points = ktk.read_c3d(gait_file, convert_point_unit=True)["Points"]

# Rename markers
for marker in marker_rename:
    if marker in gait_points.data:
        gait_points.rename_data(marker, marker_rename[marker], in_place=True)

# Add the joint centres
tracked_points = ktk.kinematics.track_cluster(gait_points, clusters["Pelvis"])
gait_points.add_data("RHJC", tracked_points.data["RHJC"], in_place=True)
gait_points.add_data("LHJC", tracked_points.data["LHJC"], in_place=True)

tracked_points = ktk.kinematics.track_cluster(gait_points, clusters["RThigh"])
gait_points.add_data("RKJC", tracked_points.data["RKJC"], in_place=True)

tracked_points = ktk.kinematics.track_cluster(gait_points, clusters["LThigh"])
gait_points.add_data("LKJC", tracked_points.data["LKJC"], in_place=True)

tracked_points = ktk.kinematics.track_cluster(gait_points, clusters["RShank"])
gait_points.add_data("RAJC", tracked_points.data["RAJC"], in_place=True)

tracked_points = ktk.kinematics.track_cluster(gait_points, clusters["LShank"])
gait_points.add_data("LAJC", tracked_points.data["LAJC"], in_place=True)


# Create rigid bodies
gait_bodies = ktk.TimeSeries(time=gait_points.time)
gait_bodies.add_data(
    "Pelvis",
    ktk.geometry.create_transform_series(
        x=gait_points.data["RASI"] - gait_points.data["LASI"],
        xy=(gait_points.data["RASI"] + gait_points.data["LASI"])
        - (gait_points.data["RPSI"] + gait_points.data["LPSI"]),
        positions=0.5 * (gait_points.data["RPSI"] + gait_points.data["LPSI"]),
    ),
    in_place=True,
)
gait_bodies.add_data(
    "RThigh",
    ktk.geometry.create_transform_series(
        z=gait_points.data["RHJC"] - gait_points.data["RKJC"],
        xz=gait_points.data["RKNE"] - gait_points.data["RKJC"],
        positions=gait_points.data["RHJC"],
    ),
    in_place=True,
)
gait_bodies.add_data(
    "LThigh",
    ktk.geometry.create_transform_series(
        z=gait_points.data["LHJC"] - gait_points.data["LKJC"],
        xz=gait_points.data["LKJC"] - gait_points.data["LKNE"],
        positions=gait_points.data["LHJC"],
    ),
    in_place=True,
)
gait_bodies.add_data(
    "RShank",
    ktk.geometry.create_transform_series(
        z=gait_points.data["RKJC"] - gait_points.data["RAJC"],
        xz=gait_points.data["RANK"] - gait_points.data["RAJC"],
        positions=gait_points.data["RKJC"],
    ),
    in_place=True,
)
gait_bodies.add_data(
    "LShank",
    ktk.geometry.create_transform_series(
        z=gait_points.data["LKJC"] - gait_points.data["LAJC"],
        xz=gait_points.data["LAJC"] - gait_points.data["LANK"],
        positions=gait_points.data["LKJC"],
    ),
    in_place=True,
)
gait_bodies.add_data(
    "RFoot",
    ktk.geometry.create_transform_series(
        y=gait_points.data["RTOE"] - gait_points.data["RHEE"],
        xy=gait_points.data["RANK"] - gait_points.data["RAJC"],
        positions=gait_points.data["RAJC"],
    ),
    in_place=True,
)
gait_bodies.add_data(
    "LFoot",
    ktk.geometry.create_transform_series(
        y=gait_points.data["LTOE"] - gait_points.data["LHEE"],
        xy=gait_points.data["LAJC"] - gait_points.data["LANK"],
        positions=gait_points.data["LAJC"],
    ),
    in_place=True,
)


# Calculate angles
angles = ktk.TimeSeries(time=gait_points.time)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["RThigh"], gait_bodies.data["Pelvis"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("RHipFlexion", euler_angles[:, 0], in_place=True)
angles.add_data("RHipAbduction", -euler_angles[:, 1], in_place=True)
angles.add_data("RHipExtRot", -euler_angles[:, 2], in_place=True)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["LThigh"], gait_bodies.data["Pelvis"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("LHipFlexion", euler_angles[:, 0], in_place=True)
angles.add_data("LHipAbduction", euler_angles[:, 1], in_place=True)
angles.add_data("LHipExtRot", euler_angles[:, 2], in_place=True)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["RShank"], gait_bodies.data["RThigh"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("RKneeFlexion", -euler_angles[:, 0], in_place=True)
angles.add_data("RKneeAbduction", -euler_angles[:, 1], in_place=True)
angles.add_data("RKneeExtRot", -euler_angles[:, 2], in_place=True)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["LShank"], gait_bodies.data["LThigh"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("LKneeFlexion", -euler_angles[:, 0], in_place=True)
angles.add_data("LKneeAbduction", euler_angles[:, 1], in_place=True)
angles.add_data("LKneeExtRot", euler_angles[:, 2], in_place=True)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["RFoot"], gait_bodies.data["RShank"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("RAnkleFlexion", euler_angles[:, 0], in_place=True)
angles.add_data("RAnkleAbduction", -euler_angles[:, 1], in_place=True)
angles.add_data("RAnkleExtRot", -euler_angles[:, 2], in_place=True)

euler_angles = ktk.geometry.get_angles(
    ktk.geometry.get_local_coordinates(
        gait_bodies.data["LFoot"], gait_bodies.data["LShank"]
    ),
    seq="XYZ",
    degrees=True,
)
angles.add_data("LAnkleFlexion", euler_angles[:, 0], in_place=True)
angles.add_data("LAnkleAbduction", euler_angles[:, 1], in_place=True)
angles.add_data("LAnkleExtRot", euler_angles[:, 2], in_place=True)


# Reference angles
angles.add_data(
    "RHipFlexion_control",
    1000 * gait_points.data["RHipAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "RHipAbduction_control",
    -1000 * gait_points.data["RHipAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "RHipExtRot_control",
    -1000 * gait_points.data["RHipAngles"][:, 2],
    in_place=True,
)

angles.add_data(
    "LHipFlexion_control",
    1000 * gait_points.data["LHipAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "LHipAbduction_control",
    -1000 * gait_points.data["LHipAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "LHipExtRot_control",
    -1000 * gait_points.data["LHipAngles"][:, 2],
    in_place=True,
)

angles.add_data(
    "RKneeFlexion_control",
    1000 * gait_points.data["RKneeAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "RKneeAbduction_control",
    -1000 * gait_points.data["RKneeAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "RKneeExtRot_control",
    -1000 * gait_points.data["RKneeAngles"][:, 2],
    in_place=True,
)

angles.add_data(
    "LKneeFlexion_control",
    1000 * gait_points.data["LKneeAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "LKneeAbduction_control",
    -1000 * gait_points.data["LKneeAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "LKneeExtRot_control",
    -1000 * gait_points.data["LKneeAngles"][:, 2],
    in_place=True,
)

angles.add_data(
    "RAnkleFlexion_control",
    1000 * gait_points.data["RAnkleAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "RAnkleAbduction_control",
    -1000 * gait_points.data["RAnkleAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "RAnkleExtRot_control",
    -1000 * gait_points.data["RAnkleAngles"][:, 2],
    in_place=True,
)

angles.add_data(
    "LAnkleFlexion_control",
    1000 * gait_points.data["LAnkleAngles"][:, 0],
    in_place=True,
)
angles.add_data(
    "LAnkleAbduction_control",
    -1000 * gait_points.data["LAnkleAngles"][:, 1],
    in_place=True,
)
angles.add_data(
    "LAnkleExtRot_control",
    -1000 * gait_points.data["LAnkleAngles"][:, 2],
    in_place=True,
)


ktk.Player(
    gait_points,
    gait_bodies,
    interconnections=interconnections_cgm,
    up="z",
    anterior="y",
)

plt.figure()
plt.subplot(3, 3, 1)
angles.plot(
    [
        "RHipFlexion",
        "LHipFlexion",
        "RHipFlexion_control",
        "LHipFlexion_control",
    ]
)
plt.subplot(3, 3, 2)
angles.plot(
    [
        "RHipAbduction",
        "LHipAbduction",
        "RHipAbduction_control",
        "LHipAbduction_control",
    ]
)
plt.subplot(3, 3, 3)
angles.plot(
    [
        "RHipExtRot",
        "LHipExtRot",
        "RHipExtRot_control",
        "LHipExtRot_control",
    ]
)
plt.subplot(3, 3, 4)
angles.plot(
    [
        "RKneeFlexion",
        "LKneeFlexion",
        "RKneeFlexion_control",
        "LKneeFlexion_control",
    ]
)
plt.subplot(3, 3, 5)
angles.plot(
    [
        "RKneeAbduction",
        "LKneeAbduction",
        "RKneeAbduction_control",
        "LKneeAbduction_control",
    ]
)
plt.subplot(3, 3, 6)
angles.plot(
    [
        "RKneeExtRot",
        "LKneeExtRot",
        "RKneeExtRot_control",
        "LKneeExtRot_control",
    ]
)
plt.subplot(3, 3, 7)
angles.plot(
    [
        "RAnkleFlexion",
        "LAnkleFlexion",
        "RAnkleFlexion_control",
        "LAnkleFlexion_control",
    ]
)
plt.subplot(3, 3, 8)
angles.plot(
    [
        "RAnkleAbduction",
        "LAnkleAbduction",
        "RAnkleAbduction_control",
        "LAnkleAbduction_control",
    ]
)
plt.subplot(3, 3, 9)
angles.plot(
    [
        "RAnkleExtRot",
        "LAnkleExtRot",
        "RAnkleExtRot_control",
        "LAnkleExtRot_control",
    ]
)

plt.tight_layout()
