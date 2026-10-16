# SPDX-License-Identifier: AGPL-3.0-only
"""EIEM's Tda Miku reference, expressed in Blender armature coordinates.

Adapted from Sasye/EIEM direct_vmd_source.h and direct_vmd_pose.h at
1bc9baa15df9eef712e53c9ee57efc7c14cf8436. See THIRD_PARTY_NOTICES.md.
No scene objects or persistent caches are created for this reference.
"""

import numpy as np


MMD_TO_BLENDER = np.array([[1., 0, 0], [0, 0, 1], [0, 1, 0]])


def tda_reference():
    def convert(v):
        return MMD_TO_BLENDER @ np.asarray(v, dtype=np.float64)

    directions = {role: convert(v) for role, v in {
        "spine": (0, .999, .044), "chest": (0, .999, .044),
        "neck": (0, 1, -.0099),
    }.items()}
    wrists = {}
    for side, sign in (("L", 1), ("R", -1)):
        for role, v in {
            "clavicle": (.9687, -.248, .0138),
            "arm": (.7941, -.6076, .012),
            "elbow": (.7962, -.6047, -.0182),
            "hand": (.79991301, -.59715137, -.05957701),
        }.items():
            directions[f"{side}_{role}"] = convert((v[0]*sign, v[1], v[2]))
        wrists[side] = (convert((.721561*sign, -.538660, -.0537414)),
                        convert((.107932*sign, -.03325, -.5565567)))
    return {"name": "Tda Miku (EIEM)", "leg_length": 10.62420198,
            "lower_from_center": convert((0, 4.74919, -.51217)),
            "groove_from_center": convert((0, .2, 0)),
            "foot_from_parent": convert((0, .79506, 0)),
            "directions": directions, "wrists": wrists}
