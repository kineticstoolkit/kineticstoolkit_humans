#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Provide tools associated to anthropometric measurements and estimates.

Warning
-------
This code is in very early development and everything can still change.
Please don't use this module in production code.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.lab as ktk
from kineticstoolkit.typing_ import ArrayLike
import numpy as np
import pandas as pd
from io import StringIO

# %% Tables


def _read_markdown_table(text: str) -> pd.DataFrame:
    """Read a markdown table."""
    return (
        pd.read_csv(
            StringIO(text.replace(" ", "")),  # Get rid of whitespaces
            sep="|",
        )
        .dropna(axis=1, how="all")
        .iloc[1:]
    )


# fmt: off
INERTIA = _read_markdown_table("""
| Segment   | Source       | LengthPoint1                | LengthPoint2                | Origin                      | Gender | Length | RelMass | RelComX | RelComY | RelComZ | RelIXX | RelIYY | RelIZZ | RelIXY | RelIXZ | RelIYZ |
|:----------|:-------------|:----------------------------|:----------------------------|:----------------------------|:-------|-------:|--------:|--------:|--------:|--------:|-------:|-------:|-------:|-------:|:------:|-------:|
| HeadNeck  | Dumas2007    | C7T1JointCenter             | HeadVertex                  | C7T1JointCenter             | F      |  0.253 |   0.067 |   0.016 |   0.575 |   0.001 |   0.29 |   0.23 |   0.30 | 0.04j  | 0.01j  | 0.00   |
| HeadNeck  | Dumas2007    | C7T1JointCenter             | HeadVertex                  | C7T1JointCenter             | M      |  0.277 |   0.067 |   0.02  |   0.536 |   0.001 |   0.28 |   0.21 |   0.30 | 0.07j  | 0.02j  | 0.03   |
| Thorax    | Dumas2007    | L5S1JointCenter             | C7T1JointCenter             | C7T1JointCenter             | F      |  0.429 |   0.304 |  -0.016 |  -0.436 |  -0.006 |   0.29 |   0.27 |   0.29 | 0.22   | 0.05   | 0.05j  |
| Thorax    | Dumas2007    | L5S1JointCenter             | C7T1JointCenter             | C7T1JointCenter             | M      |  0.477 |   0.333 |  -0.036 |  -0.42  |  -0.002 |   0.27 |   0.25 |   0.28 | 0.18   | 0.02   | 0.04j  |
| Thorax    | Dumas2007Alt | SupraSternale               | C7T1JointCenter             | SupraSternale               | F      |  0.125 |   0.304 |  -0.411 |  -1.173 |  -0.019 |   0.98 |   0.93 |   0.98 | 0.76   | 0.16   | 0.19j  |
| Thorax    | Dumas2007Alt | SupraSternale               | C7T1JointCenter             | SupraSternale               | M      |  0.139 |   0.333 |  -0.456 |  -1.121 |  -0.008 |   0.93 |   0.85 |   0.96 | 0.62   | 0.07   | 0.13j  |
| Arm       | Dumas2007    | GlenohumeralJointCenter     | ElbowJointCenter            | GlenohumeralJointCenter     | F      |  0.243 |   0.022 |  -0.073 |  -0.454 |  -0.028 |   0.33 |   0.17 |   0.33 | 0.03   | 0.05j  | 0.14   |
| Arm       | Dumas2007    | GlenohumeralJointCenter     | ElbowJointCenter            | GlenohumeralJointCenter     | M      |  0.271 |   0.024 |   0.017 |  -0.452 |  -0.026 |   0.31 |   0.14 |   0.32 | 0.06   | 0.05   | 0.02   |
| Forearm   | Dumas2007    | ElbowJointCenter            | WristJointCenter            | ElbowJointCenter            | F      |  0.247 |   0.013 |   0.021 |  -0.411 |   0.019 |   0.26 |   0.14 |   0.25 | 0.10   | 0.04   | 0.13j  |
| Forearm   | Dumas2007    | ElbowJointCenter            | WristJointCenter            | ElbowJointCenter            | M      |  0.253 |   0.017 |   0.010 |  -0.417 |   0.014 |   0.28 |   0.11 |   0.27 | 0.03   | 0.02   | 0.08j  |
| Hand      | Dumas2007    | WristJointCenter            | CarpalMetaHeadM25           | WristJointCenter            | F      |  0.071 |   0.005 |   0.077 |  -0.768 |   0.048 |   0.63 |   0.43 |   0.58 | 0.29   | 0.23   | 0.28j  |
| Hand      | Dumas2007    | WristJointCenter            | CarpalMetaHeadM25           | WristJointCenter            | M      |  0.080 |   0.006 |   0.082 |  -0.839 |   0.074 |   0.61 |   0.38 |   0.56 | 0.22   | 0.15   | 0.20j  |
| Hand      | Dumas2007Alt | WristJointCenter            | FingerTip3                  | WristJointCenter            | F      |  0.167 |   0.005 |   0.033 |  -0.327 |   0.021 |   0.27 |   0.18 |   0.25 | 0.12   | 0.1j   | 0.12j  |
| Hand      | Dumas2007Alt | WristJointCenter            | FingerTip3                  | WristJointCenter            | M      |  0.189 |   0.006 |   0.035 |  -0.357 |   0.032 |   0.26 |   0.16 |   0.24 | 0.09   | 0.07   | 0.08j  |
| Pelvis    | Dumas2007    | L5S1JointCenter             | ProjectedHipJointCenter     | L5S1JointCenter             | F      |  0.107 |   0.146 |  -0.009 |  -0.232 |   0.002 |   0.91 |   1.00 |   0.79 | 0.34j  | 0.01j  | 0.01j  |
| Pelvis    | Dumas2007    | L5S1JointCenter             | ProjectedHipJointCenter     | L5S1JointCenter             | M      |  0.094 |   0.142 |   0.028 |  -0.28  |  -0.006 |   1.01 |   1.06 |   0.95 | 0.25j  | 0.12j  | 0.08j  |
| Pelvis    | Dumas2007Alt | AnteriorSuperiorIliacSpineR | AnteriorSuperiorIliacSpineL | AnteriorSuperiorIliacSpineM | F      |  0.238 |   0.146 |  -0.371 |  -0.05  |   0.001 |   0.41 |   0.45 |   0.36 | 0.15j  | 0.00   | 0.00   |
| Pelvis    | Dumas2007Alt | AnteriorSuperiorIliacSpineR | AnteriorSuperiorIliacSpineL | AnteriorSuperiorIliacSpineM | M      |  0.224 |   0.142 |  -0.336 |  -0.149 |  -0.003 |   0.42 |   0.44 |   0.4  | 0.10j  | 0.05j  | 0.03j  |
| Thigh     | Dumas2007    | HipJointCenter              | KneeJointCenter             | HipJointCenter              | F      |  0.379 |   0.146 |  -0.077 |  -0.377 |   0.009 |   0.31 |   0.19 |   0.32 | 0.07   | 0.02j  | 0.07j  |
| Thigh     | Dumas2007    | HipJointCenter              | KneeJointCenter             | HipJointCenter              | M      |  0.432 |   0.123 |  -0.041 |  -0.429 |   0.033 |   0.29 |   0.15 |   0.3  | 0.07   | 0.02j  | 0.07j  |
| Leg       | Dumas2007    | KneeJointCenter             | AnkleJointCenter            | KneeJointCenter             | F      |  0.388 |   0.045 |  -0.049 |  -0.404 |   0.031 |   0.28 |   0.1  |   0.28 | 0.02   | 0.01   | 0.06   |
| Leg       | Dumas2007    | KneeJointCenter             | AnkleJointCenter            | KneeJointCenter             | M      |  0.433 |   0.045 |  -0.048 |  -0.41  |   0.007 |   0.28 |   0.1  |   0.28 | 0.04j  | 0.02j  | 0.05   |
| Foot      | Dumas2007    | AnkleJointCenter            | TarsalMetaHeadM15           | AnkleJointCenter            | F      |  0.165 |   0.01  |   0.270 |  -0.218 |   0.039 |   0.17 |   0.36 |   0.35 | 0.1j   | 0.06   | 0.04j  |
| Foot      | Dumas2007    | AnkleJointCenter            | TarsalMetaHeadM15           | AnkleJointCenter            | M      |  0.183 |   0.012 |   0.382 |  -0.151 |   0.026 |   0.17 |   0.37 |   0.36 | 0.13   | 0.08j  | 0.00   |
| Foot      | Dumas2007Alt | Calcaneum                   | ToeTip1                     | Calcaneum                   | F      |  0.233 |   0.01  |   0.443 |   0.044 |  -0.025 |   0.12 |   0.25 |   0.25 | 0.07j  | 0.05   | 0.03j  |
| Foot      | Dumas2007Alt | Calcaneum                   | ToeTip1                     | Calcaneum                   | M      |  0.265 |   0.012 |   0.436 |  -0.025 |  -0.007 |   0.11 |   0.25 |   0.25 | 0.09   | 0.06j  | 0.00   |
""")

# fmt: on


# %% Infer joint centers


def _infer_hip_l5s1_centers(
    *,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
    sym: ArrayLike,
    sex: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right hip, left hip, l5s1."""
    # Create a local coordinate system at the anterior superior iliac spines
    # midpoint, according to Reed et al.
    masis = 0.5 * (rasis + lasis)
    lcs = ktk.geometry.create_frames(
        origin=masis, y=lasis - rasis, yz=masis - sym
    )

    # Calculate the points in the local coordinate system
    local_rasis = ktk.geometry.get_local_coordinates(rasis, lcs)
    local_lasis = ktk.geometry.get_local_coordinates(lasis, lcs)
    local_rpsis = ktk.geometry.get_local_coordinates(rpsis, lcs)
    local_lpsis = ktk.geometry.get_local_coordinates(lpsis, lcs)
    local_sym = ktk.geometry.get_local_coordinates(sym, lcs)

    # Create a cluster using these locations
    cluster = {
        "rasis": np.nanmean(local_rasis, axis=0)[np.newaxis],
        "lasis": np.nanmean(local_lasis, axis=0)[np.newaxis],
        "rpsis": np.nanmean(local_rpsis, axis=0)[np.newaxis],
        "lpsis": np.nanmean(local_lpsis, axis=0)[np.newaxis],
        "sym": np.nanmean(local_sym, axis=0)[np.newaxis],
    }

    points = ktk.TimeSeries(
        time=np.arange(len(rasis)),
        data={"rasis": rasis, "lasis": lasis, "lpsis": lpsis, "sym": sym},
    )

    # Track the pelvis using this definition
    tracked_pelvis = ktk.kinematics.track_cluster(
        points, cluster, include_lcs=True, lcs_name="pelvis_lcs"
    )

    # Pelvis width
    pw = np.abs(np.nanmean(local_rasis[:, 1] - local_lasis[:, 1]))

    # L5S1
    if sex == "F":
        local_position = np.array([[-0.289 * pw, 0.0, 0.172 * pw, 1.0]])
    elif sex == "M":
        local_position = np.array([[-0.264 * pw, 0.0, 0.126 * pw, 1.0]])
    else:
        raise ValueError("sex must be either 'M' or 'F'")

    l5s1 = ktk.geometry.get_global_coordinates(
        local_position, tracked_pelvis.data["pelvis_lcs"]
    )

    # Right hip joint center
    if sex == "F":
        local_position = np.array(
            [[-0.197 * pw, -0.372 * pw, -0.270 * pw, 1.0]]
        )
    else:  # M
        local_position = np.array(
            [[-0.208 * pw, -0.361 * pw, -0.278 * pw, 1.0]]
        )
    rhip = ktk.geometry.get_global_coordinates(
        local_position, tracked_pelvis.data["pelvis_lcs"]
    )

    # Left joint center
    if sex == "F":
        local_position = np.array(
            [[-0.197 * pw, 0.372 * pw, -0.270 * pw, 1.0]]
        )
    else:  # M
        local_position = np.array(
            [[-0.208 * pw, 0.361 * pw, -0.278 * pw, 1.0]]
        )
    lhip = ktk.geometry.get_global_coordinates(
        local_position, tracked_pelvis.data["pelvis_lcs"]
    )
    return (rhip, lhip, l5s1)


def infer_hip_joint_center_reed1999(
    *,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
    sym: ArrayLike,
    sex: str,
    side: str = "R",
) -> np.ndarray:
    """
    Infer hip joint centre based on pelvis regression (Reed et al., 1999).

    Reed, M., Manary, M.A., Schneider, L.W., 1999. Methods for Measuring and
    Representing Automobile Occupant Posture. Presented at the International
    Congress &  Exposition, pp. 1999-01–0959.
    https://doi.org/10.4271/1999-01-0959

    """
    if side == "R":
        return _infer_hip_l5s1_centers(
            rasis=rasis,
            lasis=lasis,
            rpsis=rpsis,
            lpsis=lpsis,
            sym=sym,
            sex=sex,
        )[0]
    elif side == "L":
        return _infer_hip_l5s1_centers(
            rasis=rasis,
            lasis=lasis,
            rpsis=rpsis,
            lpsis=lpsis,
            sym=sym,
            sex=sex,
        )[1]
    else:
        raise ValueError("Side must be either 'R' or 'L'.")


def infer_hip_joint_center_hara2016(
    *,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
    l_leg: float,
    side: str = "R",
) -> np.ndarray:
    """
    Infer hip joint center based on Hara et al., 2016.

    Hara, R., McGinley, J., Briggs, C., Baker, R., Sangeux, M., 2016.
    Predicting the location of the hip joint centres, impact of age group and
    sex. Sci Rep 6, 37707. https://doi.org/10.1038/srep37707

    """
    hjc_x = 0.011 - 0.063 * l_leg
    hjc_y = 0.008 + 0.086 * l_leg
    hjc_z = -0.009 - 0.078 * l_leg

    if side == "R":
        local_hip_center = [[hjc_x, hjc_y, -hjc_z, 1]]
    elif side == "L":
        local_hip_center = [[hjc_x, -hjc_y, -hjc_z, 1]]
    else:
        raise ValueError("Side must be either 'R' or 'L'.")

    return ktk.geometry.get_global_coordinates(
        local_hip_center,
        create_pelvis_lcs_davis1991(
            rasis=rasis, lasis=lasis, rpsis=rpsis, lpsis=lpsis
        ),
    )


def infer_knee_joint_center_davis1991(
    *,
    hjc: ArrayLike,
    lateral_ep: ArrayLike,
    thigh_marker: ArrayLike,
    knee_width: float,
    marker_radius: float,
    side: str = "R",
) -> np.ndarray:
    """
    Infer knee joint centre based on the CGM without knee alignment device.

    Davis, R.B., Õunpuu, S., Tyburski, D., Gage, J.R., 1991. A gait analysis
    data collection and reduction technique. Human Movement Science 10,
    575–587. https://doi.org/10.1016/0167-9457(91)90046-Z

    """
    if side == "R":
        s = 1
    elif side == "L":
        s = -1
    else:
        raise ValueError("Side must be either 'R' or 'L'.")
    local_knee_joint_center = [
        [0, -s * (marker_radius + 0.5 * knee_width), 0, 1]
    ]
    return ktk.geometry.get_global_coordinates(
        local_knee_joint_center,
        create_thigh_lcs_davis1991(
            hjc=hjc,
            lateral_ep=lateral_ep,
            thigh_marker=thigh_marker,
            side=side,
        ),
    )


def infer_ankle_joint_center_davis1991(
    *,
    kjc: ArrayLike,
    lateral_mal: ArrayLike,
    shank_marker: ArrayLike,
    ankle_width: float,
    marker_radius: float,
    side: str = "R",
) -> np.ndarray:
    """
    Infer knee joint center based on knee width measurement.

    Davis, R.B., Õunpuu, S., Tyburski, D., Gage, J.R., 1991. A gait analysis
    data collection and reduction technique. Human Movement Science 10,
    575–587. https://doi.org/10.1016/0167-9457(91)90046-Z

    """
    if side == "R":
        s = 1
    elif side == "L":
        s = -1
    else:
        raise ValueError("Side must be either 'R' or 'L'.")
    local_ankle_joint_center = [
        [0, -s * (marker_radius + 0.5 * ankle_width), 0, 1]
    ]
    return ktk.geometry.get_global_coordinates(
        local_ankle_joint_center,
        create_shank_lcs_davis1991(
            kjc=kjc,
            lateral_mal=lateral_mal,
            shank_marker=shank_marker,
            side=side,
        ),
    )


def infer_l5s1_joint_center(
    *,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
    sym: ArrayLike,
    sex: str,
    side: str = "R",
) -> np.ndarray:
    """
    Infer L5-S1 joint center based on pelvis regression (Reed et al., 1999).

    Reed, M., Manary, M.A., Schneider, L.W., 1999. Methods for Measuring and
    Representing Automobile Occupant Posture. Presented at the International
    Congress &  Exposition, pp. 1999-01–0959.
    https://doi.org/10.4271/1999-01-0959

    """
    return _infer_hip_l5s1_centers(
        rasis=rasis, lasis=lasis, rpsis=rpsis, lpsis=lpsis, sym=sym, sex=sex
    )[2]


def infer_c7t1_joint_center(
    c7: ArrayLike,
    l5s1: ArrayLike,
    sup: ArrayLike,
    rac: ArrayLike,
    lac: ArrayLike,
    sex: str,
) -> np.ndarray:
    """
    Infer C7-T1 joint center based on thorax regression (Dumas et al., 2018).

    C7T1 is inferred using Dumas, R., Wojtusch, J., 2018. Estimation of the
    Body Segment Inertial Parameters for the Rigid Body Biomechanical Models
    Used in Motion Analysis, in: Handbook of Human Motion. Springer
    International Publishing, Cham, pp. 47–77.
    https://doi.org/10.1007/978-3-319-14418-4_147

    The trunk must be in neutral position.

    """
    # Create reference frames with x: C7-SUP, y: L5S1-C7, z: right
    c7sup = sup - c7
    thorax_lcs = ktk.geometry.create_transform_series(
        positions=c7, x=(c7sup), xy=(c7 - l5s1)
    )

    # Thorax width (tw)
    tw = np.mean(np.sqrt(np.sum((sup - c7) ** 2, axis=1)))

    if sex == "M":
        c7t1_angle = 8  # deg
        c7t1_ratio = 0.55
    elif sex == "F":
        c7t1_angle = 14  # deg
        c7t1_ratio = 0.53
    else:
        raise ValueError("sex must be 'M' or 'F'")

    local_c7t1 = ktk.geometry.create_point_series(
        x=[c7t1_ratio * tw * np.cos(np.deg2rad(c7t1_angle))],
        y=[c7t1_ratio * tw * np.sin(np.deg2rad(c7t1_angle))],
    )

    return ktk.geometry.get_global_coordinates(local_c7t1, thorax_lcs)


def infer_gh_joint_center(
    c7: ArrayLike,
    l5s1: ArrayLike,
    sup: ArrayLike,
    rac: ArrayLike,
    lac: ArrayLike,
    sex: str,
    side: str = "R",
) -> np.ndarray:
    """
    Infer glenohumeral joint center based on thorax regression (Rab et al., 2002).

    GH joint centres are inferred using Rab, G., Petuskey, K., Bagley, A.,
    2002. A method for determination of upper extremity ktk.kinematics.
    Gait & Posture 15, 113–119. https://doi.org/10.1016/S0966-6362(01)00155-2

    Arms and trunk must be in neutral position.

    """
    # Create reference frames with x: C7-SUP, y: L5S1-C7, z: right
    c7sup = sup - c7

    if side == "R":
        thorax_lcs = ktk.geometry.create_transform_series(
            positions=rac, x=(c7sup), xy=(c7 - l5s1)
        )
    elif side == "L":
        thorax_lcs = ktk.geometry.create_transform_series(
            positions=lac, x=(c7sup), xy=(c7 - l5s1)
        )

    # Interacromial distance (aw)
    ad = np.mean(np.sqrt(np.sum((rac - lac) ** 2, axis=1)))

    local_gh = np.array(
        [
            [
                0,
                -0.17 * ad,
                0.0,
                1.0,
            ]
        ]
    )

    return ktk.geometry.get_global_coordinates(local_gh, thorax_lcs)


# %% Create local coordinate systems


def create_pelvis_lcs_isb(
    *,
    l5s1: ArrayLike,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
) -> np.ndarray:
    mpsis = 0.5 * (rpsis + lpsis)
    return ktk.geometry.create_transform_series(
        positions=l5s1,
        z=(rasis - lasis),
        xz=(0.5 * (rasis + lasis) - mpsis),
    )


def create_pelvis_lcs_davis1991(
    *,
    rasis: ArrayLike,
    lasis: ArrayLike,
    rpsis: ArrayLike,
    lpsis: ArrayLike,
) -> np.ndarray:
    """
    Create Pelvis LCS based on the conventional gait model.

    Davis, R.B., Õunpuu, S., Tyburski, D., Gage, J.R., 1991. A gait analysis
    data collection and reduction technique. Human Movement Science 10,
    575–587. https://doi.org/10.1016/0167-9457(91)90046-Z

    Warning
    -------
    This paper shows two different coordinate systems for the pelvis. This
    one is Figure 3 (used to infer the hip joint centers), not Figure 2
    (used to calculate joint angles).

    """
    return ktk.geometry.create_transform_series(
        positions=0.5 * (rasis + lasis),
        y=rasis - lasis,
        xy=0.5 * (rasis + lasis) - 0.5 * (rpsis + lpsis),
    )


def create_thorax_lcs(
    *, c7t1: ArrayLike, l5s1: ArrayLike, sup: ArrayLike
) -> np.ndarray:
    return ktk.geometry.create_transform_series(
        positions=c7t1,
        y=(c7t1 - l5s1),
        xy=(sup - l5s1),
    )


def create_head_neck_lcs(
    *, c7t1: ArrayLike, hv: ArrayLike, sel: ArrayLike
) -> np.ndarray:
    return ktk.geometry.create_transform_series(
        positions=c7t1,
        y=(hv - c7t1),
        xy=(sel - c7t1),
    )


def create_arm_lcs(
    *, gh: ArrayLike, lat_ep: ArrayLike, med_ep: ArrayLike, side: str = "R"
) -> np.ndarray:
    elbow_center = 0.5 * (lat_ep + med_ep)
    if side == "R":
        return ktk.geometry.create_transform_series(
            origin=gh,
            y=(gh - elbow_center),
            yz=(lat_ep - med_ep),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            origin=gh,
            y=(gh - elbow_center),
            yz=(med_ep - lat_ep),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_forearm_lcs(
    *,
    elbow_center: ArrayLike,
    ulnar_st: ArrayLike,
    radial_st: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    wrist_center = 0.5 * (ulnar_st + radial_st)
    if side == "R":
        return ktk.geometry.create_transform_series(
            origin=elbow_center,
            y=(elbow_center - wrist_center),
            yz=(radial_st - ulnar_st),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            origin=elbow_center,
            y=(elbow_center - wrist_center),
            yz=(ulnar_st - radial_st),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_hand_lcs(
    *,
    wrist_center: ArrayLike,
    meta_head2: ArrayLike,
    meta_head5: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    meta_center = 0.5 * (meta_head2 + meta_head5)
    if side == "R":
        return ktk.geometry.create_transform_series(
            positions=wrist_center,
            y=(wrist_center - meta_center),
            yz=(meta_head2 - meta_head5),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            positions=wrist_center,
            y=(wrist_center - meta_center),
            yz=(meta_head5 - meta_head2),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_thigh_lcs_isb(
    *,
    hip_center: ArrayLike,
    lateral_ep: ArrayLike,
    medial_ep: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    knee_center = 0.5 * (lateral_ep + medial_ep)
    if side == "R":
        return ktk.geometry.create_transform_series(
            positions=hip_center,
            y=(hip_center - knee_center),
            yz=(lateral_ep - medial_ep),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            positions=hip_center,
            y=(hip_center - knee_center),
            yz=(medial_ep - lateral_ep),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_thigh_lcs_davis1991(
    *,
    hjc: ArrayLike,
    lateral_ep: ArrayLike,
    thigh_marker: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    """
    Create Thigh LCS based on the conventional gait model.

    This is the coordinate system used to infer the knee center, not the
    one to calculate joint angles.

    Davis, R.B., Õunpuu, S., Tyburski, D., Gage, J.R., 1991. A gait analysis
    data collection and reduction technique. Human Movement Science 10,
    575–587. https://doi.org/10.1016/0167-9457(91)90046-Z

    """
    if side == "R":
        return ktk.geometry.create_transform_series(
            positions=lateral_ep,
            z=hjc - lateral_ep,
            yz=thigh_marker - lateral_ep,
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            positions=lateral_ep,
            z=hjc - lateral_ep,
            yz=-(thigh_marker - lateral_ep),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_shank_lcs_davis1991(
    *,
    kjc: ArrayLike,
    lateral_mal: ArrayLike,
    shank_marker: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    """
    Create Shank LCS based on the conventional gait model.

    This is the coordinate system used to infer the ankle center, not the
    one to calculate joint angles.

    Davis, R.B., Õunpuu, S., Tyburski, D., Gage, J.R., 1991. A gait analysis
    data collection and reduction technique. Human Movement Science 10,
    575–587. https://doi.org/10.1016/0167-9457(91)90046-Z

    """
    if side == "R":
        return ktk.geometry.create_transform_series(
            positions=lateral_mal,
            z=kjc - lateral_mal,
            yz=shank_marker - lateral_mal,
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            positions=lateral_mal,
            z=kjc - lateral_mal,
            yz=-(shank_marker - lateral_mal),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_shank_lcs(
    *,
    knee_center: ArrayLike,
    lateral_mal: ArrayLike,
    medial_mal: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    ankle_center = 0.5 * (lateral_mal + medial_mal)
    if side == "R":
        return ktk.geometry.create_transform_series(
            positions=knee_center,
            y=(knee_center - ankle_center),
            yz=(lateral_mal - medial_mal),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            positions=knee_center,
            y=(knee_center - ankle_center),
            yz=(medial_mal - lateral_mal),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


def create_foot_lcs(
    *,
    ankle_center: ArrayLike,
    calc: ArrayLike,
    meta_head1: ArrayLike,
    meta_head5: ArrayLike,
    side: str = "R",
) -> np.ndarray:
    meta_center = 0.5 * (meta_head1 + meta_head5)
    if side == "R":
        return ktk.geometry.create_transform_series(
            origin=ankle_center,
            x=(meta_center - calc),
            xz=(meta_head5 - meta_head1),
        )
    elif side == "L":
        return ktk.geometry.create_transform_series(
            origin=ankle_center,
            x=(meta_center - calc),
            xz=(meta_head1 - meta_head5),
        )
    else:
        raise ValueError("Side must be either 'R' or 'L'")


# %% Inertial properties


def estimate_center_of_mass(
    points: ktk.TimeSeries,
    /,
    segments: str | list[str],
    *,
    sex: str = "M",
) -> ktk.TimeSeries:
    """
    Estimate the segments' center of mass based on anthropometric data.

    Based on

    Dumas, R., Chèze, L., Verriest, J.-P., 2007. Adjustments to McConville et al. and
    Young et al. body segment inertial parameters. Journal of Biomechanics 40, 543–553.
    https://doi.org/10.1016/j.jbiomech.2006.02.013

    and

    Dumas, R., Chèze, L., Verriest, J.-P., 2007. Corrigendum to “Adjustments to
    McConville et al. and Young et al. body segment inertial parameters” [J. Biomech.
    40 (2007) 543–553]. Journal of Biomechanics 40, 1651–1652.
    https://doi.org/10.1016/j.jbiomech.2006.07.016


    Parameters
    ----------
    points
        ktk.TimeSeries that contains marker trajectories as Nx4 during an action.

    segments
        Name of the segment. Can be 'Pelvis', 'Thorax', 'HeadNeck',
        'ArmR', 'ArmL', 'ForearmR', 'ForearmL', 'HandR', 'HandL', 'ThighR',
        'ThighL', 'LegR', 'LegL', 'FootR', or 'FootL'. An empty string or
        an empty list will try to process every segment.

    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    ktk.TimeSeries
        A ktk.TimeSeries with the trajectory of the segments' centers of mass,
        named {segment}CenterOfMass.

    """
    points = points.copy()  # We will add points to it so copy it first.
    output = points.copy(copy_data=False, copy_data_info=False)

    # If we have no segment or an empty list, go with every segment
    if len(segments) == 0:
        segments = [
            "Pelvis",
            "Thorax",
            "HeadNeck",
            "ArmR",
            "ArmL",
            "ForearmR",
            "ForearmL",
            "HandR",
            "HandL",
            "ThighR",
            "ThighL",
            "LegR",
            "LegL",
            "FootR",
            "FootL",
        ]

    # If we have a list of segments
    if not isinstance(segments, str):
        for segment in segments:
            output.merge(
                estimate_center_of_mass(points, segment),
                in_place=True,
            )
        return output

    # From here we have a single segment
    segment = segments

    # Decompose segment name and side
    if segment not in ["Pelvis", "Thorax", "HeadNeck"]:
        side = segment[-1]
        segment = segment[0:-1]
    else:
        side = ""

    # Calculate the local coordinate system for this segment
    lcs = track_local_coordinate_systems(points, segments=(segment + side))

    df = INERTIAL_VALUES["Dumas2007"]
    # Search the inertial value tables for the given segment and sex
    _ = df.loc[(df["Segment"] == segment) & (df["Gender"] == sex)]
    constants = _.to_dict("records")[0]

    # Add possible missing points
    if "ProjectedHipJointCenter" in [
        constants["LengthPoint1"],
        constants["LengthPoint2"],
    ]:
        # Calculate ProjectedHipJointCenter
        local_rhip = ktk.geometry.get_local_coordinates(
            points.data["HipJointCenterR"], lcs.data["Pelvis"]
        )
        local_rhip[:, 2] = 0  # Projection in sagittal plane
        local_lhip = ktk.geometry.get_local_coordinates(
            points.data["HipJointCenterL"], lcs.data["Pelvis"]
        )
        local_lhip[:, 2] = 0  # Projection in sagittal plane
        local_hips = 0.5 * (local_rhip + local_lhip)

        points.data["ProjectedHipJointCenter"] = (
            ktk.geometry.get_global_coordinates(local_hips, lcs.data["Pelvis"])
        )

    if "CarpalMetaHeadM25" in [
        constants["LengthPoint1"],
        constants["LengthPoint2"],
    ]:
        # Calculate midpoint of carpal meat head 2 and 5
        points.data[f"CarpalMetaHeadM25{side}"] = 0.5 * (
            points.data[f"CarpalMetaHead2{side}"]
            + points.data[f"CarpalMetaHead5{side}"]
        )

    if "TarsalMetaHeadM15" in [
        constants["LengthPoint1"],
        constants["LengthPoint2"],
    ]:
        # Calculate midpoint of carpal meat head 2 and 5
        points.data[f"TarsalMetaHeadM15{side}"] = 0.5 * (
            points.data[f"TarsalMetaHead1{side}"]
            + points.data[f"TarsalMetaHead5{side}"]
        )

    # Calculate the segment length
    segment_length = np.sqrt(
        np.sum(
            np.nanmean(
                points.data[constants["LengthPoint2"] + side]
                - points.data[constants["LengthPoint1"] + side],
                axis=0,
            )
            ** 2
        )
    )

    # Add COM output
    output.data[f"{segment}{side}CenterOfMass"] = (
        ktk.geometry.get_global_coordinates(
            np.array(
                [
                    [
                        segment_length * constants["RelComX"],
                        segment_length * constants["RelComY"],
                        segment_length * constants["RelComZ"],
                        1.0,
                    ]
                ]
            ),
            lcs.data[segment + side],
        )
    )

    return output


def estimate_global_center_of_mass(
    coms: ktk.TimeSeries, /, sex: str = "M"
) -> ktk.TimeSeries:
    """
    Estimate the global center of mass.

    Parameters
    ----------
    coms
        A ktk.TimeSeries with the trajectory of every segment's center of mass.
        The segments must be in this list: 'PelvisCenterOfMass',
        'ThoraxCenterOfMass', 'HeadNeckCenterOfMass',
        'ArmRCenterOfMass', 'ArmLCenterOfMass',
        'ForearmRCenterOfMass', 'ForearmLCenterOfMass',
        'HandRCenterOfMass', 'HandLCenterOfMass',
        'ThighRCenterOfMass', 'ThighLCenterOfMass',
        'LegRCenterOfMass', 'LegLCenterOfMass',
        'FootRCenterOfMass', 'FootLCenterOfMass'.
    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    ktk.ktk.TimeSeries
        A ktk.TimeSeries with a single element named 'GlobalCenterOfMass'.

    """
    inertial_data = INERTIAL_VALUES["Dumas2007"]

    out = coms.copy(copy_data=False, copy_data_info=False)
    out.data["GlobalCenterOfMass"] = np.zeros([out.time.shape[0], 4])

    cumulative_mass = 0.0

    for data in coms.data:
        segment_name = data.replace("CenterOfMass", "")
        if segment_name not in ["Thorax", "HeadNeck", "Pelvis"]:
            # The segment name is terminated by L or R. Remove it.
            segment_name = segment_name[0:-1]

        segment_rel_mass = inertial_data.loc[
            (inertial_data["Segment"] == segment_name)
            & (inertial_data["Gender"] == sex),
            "RelMass",
        ]

        if len(segment_rel_mass) == 1:

            out.data["GlobalCenterOfMass"] += (
                float(segment_rel_mass) * coms.data[data]
            )

            cumulative_mass += float(segment_rel_mass)

    out.data["GlobalCenterOfMass"] /= cumulative_mass
    out.data["GlobalCenterOfMass"][:, 3] = 1

    return out


# %% Constants

#: A link model to help ktk.kinematics visualization
LINKS = {
    "Pelvis": {
        "Links": [
            ["AnteriorSuperiorIliacSpineR", "AnteriorSuperiorIliacSpineL"],
            ["AnteriorSuperiorIliacSpineL", "PosteriorSuperiorIliacSpineL"],
            ["PosteriorSuperiorIliacSpineL", "PosteriorSuperiorIliacSpineR"],
            ["PosteriorSuperiorIliacSpineR", "AnteriorSuperiorIliacSpineR"],
            ["AnteriorSuperiorIliacSpineR", "HipJointCenterR"],
            ["PosteriorSuperiorIliacSpineR", "HipJointCenterR"],
            ["AnteriorSuperiorIliacSpineL", "HipJointCenterL"],
            ["PosteriorSuperiorIliacSpineL", "HipJointCenterL"],
            ["AnteriorSuperiorIliacSpineR", "PubicSymphysis"],
            ["AnteriorSuperiorIliacSpineL", "PubicSymphysis"],
            ["HipJointCenterR", "PubicSymphysis"],
            ["HipJointCenterL", "PubicSymphysis"],
            ["HipJointCenterL", "HipJointCenterR"],
        ],
        "Color": [0.25, 0.5, 0.25],
    },
    "Trunk": {
        "Links": [
            ["L5S1JointCenter", "C7T1JointCenter"],
            ["C7", "GlenohumeralJointCenterR"],
            ["C7", "GlenohumeralJointCenterL"],
            ["Suprasternale", "GlenohumeralJointCenterR"],
            ["Suprasternale", "GlenohumeralJointCenterL"],
            ["C7", "AcromionR"],
            ["C7", "AcromionL"],
            ["Suprasternale", "AcromionR"],
            ["Suprasternale", "AcromionL"],
            ["GlenohumeralJointCenterR", "AcromionR"],
            ["GlenohumeralJointCenterL", "AcromionL"],
        ],
        "Color": [0.5, 0.5, 0],
    },
    "HeadNeck": {
        "Links": [
            ["Sellion", "C7T1JointCenter"],
            ["HeadVertex", "C7T1JointCenter"],
            ["Sellion", "HeadVertex"],
        ],
        "Color": [0.5, 0.5, 0.25],
    },
    "UpperArms": {
        "Links": [
            ["GlenohumeralJointCenterR", "ElbowJointCenterR"],
            ["GlenohumeralJointCenterR", "LateralHumeralEpicondyleR"],
            ["GlenohumeralJointCenterR", "MedialHumeralEpicondyleR"],
            ["LateralHumeralEpicondyleR", "MedialHumeralEpicondyleR"],
            ["GlenohumeralJointCenterL", "ElbowJointCenterL"],
            ["GlenohumeralJointCenterL", "LateralHumeralEpicondyleL"],
            ["GlenohumeralJointCenterL", "MedialHumeralEpicondyleL"],
            ["LateralHumeralEpicondyleL", "MedialHumeralEpicondyleL"],
        ],
        "Color": [0.5, 0.25, 0],
    },
    "Forearms": {
        "Links": [
            ["ElbowJointCenterR", "WristJointCenterR"],
            ["RadialStyloidR", "LateralHumeralEpicondyleR"],
            ["UlnarStyloidR", "MedialHumeralEpicondyleR"],
            ["RadialStyloidR", "UlnarStyloidR"],
            ["ElbowJointCenterL", "WristJointCenterL"],
            ["RadialStyloidL", "LateralHumeralEpicondyleL"],
            ["UlnarStyloidL", "MedialHumeralEpicondyleL"],
            ["RadialStyloidL", "UlnarStyloidL"],
        ],
        "Color": [0.5, 0, 0],
    },
    "Hands": {
        "Links": [
            ["RadialStyloidR", "CarpalMetaHead2R"],
            ["UlnarStyloidR", "CarpalMetaHead5R"],
            ["CarpalMetaHead2R", "CarpalMetaHead5R"],
            ["RadialStyloidL", "CarpalMetaHead2L"],
            ["UlnarStyloidL", "CarpalMetaHead5L"],
            ["CarpalMetaHead2L", "CarpalMetaHead5L"],
        ],
        "Color": [0.5, 0, 0.25],
    },
    "Tighs": {
        "Links": [
            ["HipJointCenterR", "KneeJointCenterR"],
            ["HipJointCenterR", "LateralFemoralEpicondyleR"],
            ["HipJointCenterR", "MedialFemoralEpicondyleR"],
            ["LateralFemoralEpicondyleR", "MedialFemoralEpicondyleR"],
            ["HipJointCenterL", "KneeJointCenterL"],
            ["HipJointCenterL", "LateralFemoralEpicondyleL"],
            ["HipJointCenterL", "MedialFemoralEpicondyleL"],
            ["LateralFemoralEpicondyleL", "MedialFemoralEpicondyleL"],
        ],
        "Color": [0, 0.5, 0.5],
    },
    "Legs": {
        "Links": [
            ["AnkleJointCenterR", "KneeJointCenterR"],
            ["LateralMalleolusR", "LateralFemoralEpicondyleR"],
            ["MedialMalleolusR", "MedialFemoralEpicondyleR"],
            ["LateralMalleolusR", "MedialMalleolusR"],
            ["AnkleJointCenterL", "KneeJointCenterL"],
            ["LateralMalleolusL", "LateralFemoralEpicondyleL"],
            ["MedialMalleolusL", "MedialFemoralEpicondyleL"],
            ["LateralMalleolusL", "MedialMalleolusL"],
        ],
        "Color": [0, 0.25, 0.5],
    },
    "Feets": {
        "Links": [
            ["CalcaneusR", "TarsalMetaHead1R"],
            ["CalcaneusR", "TarsalMetaHead5R"],
            ["MedialMalleolusR", "TarsalMetaHead1R"],
            ["LateralMalleolusR", "TarsalMetaHead5R"],
            ["CalcaneusR", "MedialMalleolusR"],
            ["CalcaneusR", "LateralMalleolusR"],
            ["TarsalMetaHead1R", "TarsalMetaHead5R"],
            ["CalcaneusL", "TarsalMetaHead1L"],
            ["CalcaneusL", "TarsalMetaHead5L"],
            ["MedialMalleolusL", "TarsalMetaHead1L"],
            ["LateralMalleolusL", "TarsalMetaHead5L"],
            ["CalcaneusL", "MedialMalleolusL"],
            ["CalcaneusL", "LateralMalleolusL"],
            ["TarsalMetaHead1L", "TarsalMetaHead5L"],
        ],
        "Color": [0.25, 0.0, 0.75],
    },
}


# Load Inertial Values
INERTIAL_VALUES = {}


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
