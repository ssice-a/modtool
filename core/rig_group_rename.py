"""Registered vertex-group conversion presets for supported game rigs."""

from __future__ import annotations

from array import array
from collections import Counter

import numpy as np

from .common import is_mesh_object
from .rename import apply_exact_name_pairs


_WEIGHT_EPSILON = 1.0e-8
_RIG_MAPPING_SPECS = {
    "AUTO_RIG_NPC": {
        "label": "auto rig -> NPC",
        "builder": "_build_game_body_mapping",
    },
    "AUTO_RIG_ZMD": {
        "label": "autorig->zmd",
        "builder": "_build_zmd_body_mapping",
    },
    "VRC_ZMD": {
        "label": "vrc->zmd",
        "builder": "_build_vrc_zmd_mapping",
    },
}

AUTO_RIG_NPC_MAPPING_ID = "AUTO_RIG_NPC"
AUTO_RIG_ZMD_MAPPING_ID = "AUTO_RIG_ZMD"
VRC_ZMD_MAPPING_ID = "VRC_ZMD"
_RIG_MAPPING_ITEMS = tuple(
    (mapping_id, spec["label"], f"Registered mapping: {spec['label']}")
    for mapping_id, spec in _RIG_MAPPING_SPECS.items()
)


def rig_mapping_items(_self, _context):
    """Return the registered source-to-target rig mapping choices."""
    return _RIG_MAPPING_ITEMS


def _target_side_name(side, part):
    return f"Bip001-{side}-{part}"


def _build_game_body_mapping(source_names, *, metacarpals_to_hand=False):
    """Build the offline rig_E-to-NPC vertex-group mapping.

    This table is deliberately independent of Blender armatures and target
    objects. It is an offline naming contract for the supported NPC body.
    """
    mapping = {}

    fixed = {
        "root.x": "Bip001-Pelvis",
        "spine_01.x": "Bip001-Spine",
        "spine_02.x": "Bip001-Spine1",
        "spine_03.x": "Bip001-Spine2",
        "spine_04.x": "Bip001-Spine2",
        "neck.x": "Bip001-Neck",
        "head.x": "Bip001-Head",
    }
    mapping.update({name: target for name, target in fixed.items() if name in source_names})

    for side in ("L", "R"):
        lower_side = side.lower()
        sided_parts = {
            f"shoulder.{lower_side}": _target_side_name(side, "Clavicle"),
            # The NPC body mesh does not weight the generic Bip001 limb
            # bones. Its four rig_E limb controls map to the two twist
            # deform bones for each segment.
            f"arm_stretch.{lower_side}": f"Bone-{side}-UpperArm-Twist1",
            f"forearm_stretch.{lower_side}": f"Bone-{side}-ForeArm-Twist",
            f"hand.{lower_side}": _target_side_name(side, "Hand"),
            f"thigh_stretch.{lower_side}": f"Bone-{side}-Thigh-Twist1",
            f"leg_stretch.{lower_side}": f"Bone-{side}-Calf-Twist",
            f"foot.{lower_side}": _target_side_name(side, "Foot"),
            f"toes_01.{lower_side}": _target_side_name(side, "Toe0"),
            f"forearm_twist.{lower_side}": f"Bone-{side}-ForeArm-Twist1",
            f"leg_twist.{lower_side}": f"Bone-{side}-Calf-Twist1",
            f"thigh_twist.{lower_side}": f"Bone-{side}-Thigh-Twist",
            f"c_arm_twist_offset.{lower_side}": f"Bone-{side}-UpperArm-Twist",
            f"thumb1.{lower_side}": _target_side_name(side, "Finger0"),
            f"c_thumb2.{lower_side}": _target_side_name(side, "Finger01"),
            f"c_thumb3.{lower_side}": _target_side_name(side, "Finger02"),
        }
        mapping.update({name: target for name, target in sided_parts.items() if name in source_names})

        finger_parts = {
            "index": 1,
            "middle": 2,
            "ring": 3,
            "pinky": 4,
        }
        for source_part, target_part in finger_parts.items():
            # ZMD has no separate metacarpal bones: these belong to Hand,
            # while the three phalanges retain three independent targets.
            finger_targets = {
                f"c_{source_part}1_base.{lower_side}": (
                    "Hand" if metacarpals_to_hand else f"Finger{target_part}"
                ),
                f"{source_part}1.{lower_side}": (
                    f"Finger{target_part}" if metacarpals_to_hand else f"Finger{target_part}1"
                ),
                f"c_{source_part}2.{lower_side}": f"Finger{target_part}1",
                f"c_{source_part}3.{lower_side}": f"Finger{target_part}2",
            }
            for source_name, target_part_name in finger_targets.items():
                if source_name in source_names:
                    mapping[source_name] = _target_side_name(side, target_part_name)

        mapping.update(
            {
                f"BigToe.{side}": _target_side_name(side, "Toe0"),
                f"BigToe_2.{side}": _target_side_name(side, "Toe0"),
            }
        )
        for toe_index in range(1, 5):
            for suffix in (f"SmallToe{toe_index}.{side}", f"SmallToe{toe_index}_2.{side}"):
                if suffix in source_names:
                    mapping[suffix] = _target_side_name(side, "Toe0")

        chest_side = "r" if side == "L" else "l"
        for source_name in (f"Breast_0.{side}", f"Breast_0_{side}"):
            if source_name in source_names:
                mapping[source_name] = f"Bn_{chest_side}_breast_001"
        for source_name in (
            f"Breast.{side}",
            f"Breast.{side}.001",
            f"Breast_{side}",
            f"breastM_{side}",
        ):
            if source_name in source_names:
                mapping[source_name] = f"Bn_{chest_side}_breast_002"

        butt_target = f"Bn_{lower_side}_butt_001"
        for source_name in (f"Butt.{side}", f"Butt_{side}"):
            if source_name in source_names:
                mapping[source_name] = butt_target

    pelvis_groups = {
        "Anus_control",
        "duzi",
        "Vagina_00.l",
        "Vagina_000.l",
        "Vagina_01.l",
        "Vagina_02.l",
        "Vagina_03.l",
        "Vagina_bone_1",
        "Vagina_bone_2",
        "Vagina_bone_3",
        "Vagina_bone_4",
        "Vagina_00.r",
        "Vagina_000.r",
        "Vagina_01.r",
        "Vagina_02.r",
        "Vagina_03.r",
    }
    for source_name in pelvis_groups:
        if source_name in source_names:
            mapping[source_name] = "Bip001-Pelvis"

    return mapping


_NPC_IDENTITY_GROUPS = {
    "Bip001-Pelvis",
    "Bip001-Spine",
    "Bip001-Spine1",
    "Bip001-Spine2",
    "Bip001-Neck",
    "Bip001-Head",
    "Bn_l_breast_001",
    "Bn_l_breast_002",
    "Bn_r_breast_001",
    "Bn_r_breast_002",
    "Bn_l_butt_001",
    "Bn_r_butt_001",
}


def _zmd_target_names():
    """Translate the shared autorig group roles to the inspected Typhoea rig."""
    targets = {
        f"Bip001-{part}": f"Bip001_{part}"
        for part in ("Pelvis", "Spine", "Spine1", "Spine2", "Neck", "Head")
    }
    for side in ("L", "R"):
        for part in ("Clavicle", "Hand", "Foot", "Toe0"):
            targets[_target_side_name(side, part)] = f"Bip001_{side}_{part}"
        for finger in range(5):
            for suffix in ("", "1", "2"):
                part = f"Finger{finger}{suffix}"
                targets[_target_side_name(side, part)] = f"Bip001_{side}_{part}"
        targets.update({
            f"Bone-{side}-UpperArm-Twist": f"Bip001_{side}UpArmTwist",
            f"Bone-{side}-UpperArm-Twist1": f"Bip001_{side}UpArmTwist1",
            f"Bone-{side}-ForeArm-Twist": f"Bip001_{side}_ForeTwist",
            f"Bone-{side}-ForeArm-Twist1": f"Bip001_{side}_ForeTwist1",
            f"Bone-{side}-Thigh-Twist": f"Bip001_{side}ThighTwist",
            f"Bone-{side}-Thigh-Twist1": f"Bip001_{side}ThighTwist1",
            f"Bone-{side}-Calf-Twist": f"Bip001_{side}CalfTwist",
            f"Bone-{side}-Calf-Twist1": f"Bip001_{side}CalfTwist1",
            f"Bn_{side.lower()}_butt_001": "Bip001_Pelvis",
        })
        for index in (1, 2):
            targets[f"Bn_{side.lower()}_breast_{index:03d}"] = (
                f"breast_base_{side}_a_01_jnt"
            )
    return targets


_ZMD_TARGET_BY_ROLE = _zmd_target_names()
_ZMD_TARGET_ALIASES = {
    "breast_base_L_a_01_jnt": ("L_breast_01_jnt", "xiong_L_0_skin_jnt"),
    "breast_base_R_a_01_jnt": ("R_breast_01_jnt", "xiong_R_0_skin_jnt"),
}
_ZMD_IDENTITY_GROUPS = frozenset(_ZMD_TARGET_BY_ROLE.values()) | {
    f"Bip001_{side}_{part}{suffix}"
    for side in ("L", "R")
    for part, suffixes in {
        "UpperArm": ("", "_corrective_ty_minus", "_corrective_ty_plus", "_corrective_tz_plus"),
        "Forearm": ("", "_corrective_ty_minus"),
        "Hand": ("_corrective_ty_minus", "_corrective_ty_plus"),
        "Thigh": ("", "_corrective_ty_minus", "_corrective_ty_plus", "_corrective_tz_minus"),
        "Calf": ("", "_corrective_ty_minus", "_corrective_ty_plus"),
        "Foot": ("_corrective_ty_minus", "_corrective_ty_plus"),
    }.items()
    for suffix in suffixes
}


def _resolve_target_name(target_name, target_bone_names):
    """Resolve a preset target to a bone present on the selected armature."""
    if target_bone_names is None or target_name in target_bone_names:
        return target_name
    for alias in _ZMD_TARGET_ALIASES.get(target_name, ()):
        if alias in target_bone_names:
            return alias
    return None


def _build_zmd_body_mapping(source_names, *, target_bone_names=None):
    # EF_Size_A's Breast.R is on +X, matching the game's breast_base_L.
    # Reuse source aliases, with ZMD's distinct palm/phalange assignment.
    mapping = {
        source: _ZMD_TARGET_BY_ROLE[role]
        for source, role in _build_game_body_mapping(
            source_names, metacarpals_to_hand=True
        ).items()
    }
    if target_bone_names is None:
        return mapping
    for source in _ZMD_TARGET_ALIASES:
        if source in source_names:
            resolved = _resolve_target_name(source, target_bone_names)
            if resolved is not None:
                mapping[source] = resolved
    return {
        source: resolved
        for source, target in mapping.items()
        for resolved in (_resolve_target_name(target, target_bone_names),)
        if resolved is not None
    }


def _vrc_zmd_targets():
    """Inspected VRC body roles; limb controls use the game's main bones."""
    targets = {
        "Hips": "Bip001_Pelvis",
        "Spine": "Bip001_Spine",
        "Chest": "Bip001_Spine2",
        "Neck": "Bip001_Neck",
        "Head": "Bip001_Head",
    }
    for side in ("L", "R"):
        for source, target in {
            "Shoulder": "Clavicle", "Arm": "UpperArm", "Elbow": "Forearm",
            "Wrist": "Hand", "Leg": "Thigh", "Knee": "Calf",
            "Foot": "Foot", "Toe": "Toe0",
            "Thumb": "Finger0", "Thumb 01": "Finger01", "Thumb 03": "Finger02",
        }.items():
            targets[f"{source}_{side}"] = f"Bip001_{side}_{target}"
        for finger, number in (("Index", 1), ("Middle", 2), ("Ring", 3), ("Little", 4)):
            # Gemma's first ring-finger bone has a space before the underscore.
            first = f"{finger} Finger{' ' if finger == 'Ring' else ''}_{side}"
            targets[first] = f"Bip001_{side}_Finger{number}"
            for segment, suffix in (("02", "1"), ("03", "2")):
                targets[f"{finger} Finger {segment}_{side}"] = (
                    f"Bip001_{side}_Finger{number}{suffix}"
                )
    return targets


_VRC_ZMD_TARGETS = _vrc_zmd_targets()


def _build_vrc_zmd_mapping(source_names):
    """Map known body names only; unmapped physics groups are left alone."""
    return {
        source: target for source, target in _VRC_ZMD_TARGETS.items()
        if source in source_names
    }


def _build_mapping(mapping_id, source_names, *, target_bone_names=None):
    spec = _RIG_MAPPING_SPECS.get(mapping_id)
    if spec is None:
        raise ValueError(f"Unknown rig mapping table: {mapping_id}")
    builder = globals()[spec["builder"]]
    builder_kwargs = {}
    if mapping_id == AUTO_RIG_ZMD_MAPPING_ID:
        builder_kwargs["target_bone_names"] = target_bone_names
    mapping = builder(source_names, **builder_kwargs)
    identity_groups = (
        _ZMD_IDENTITY_GROUPS
        if mapping_id in (AUTO_RIG_ZMD_MAPPING_ID, VRC_ZMD_MAPPING_ID)
        else _NPC_IDENTITY_GROUPS
    )
    if target_bone_names is not None:
        identity_groups = identity_groups | set(target_bone_names)
    mapping.update({name: name for name in source_names & identity_groups})
    if target_bone_names is not None:
        if mapping_id == VRC_ZMD_MAPPING_ID:
            missing = sorted(set(mapping.values()) - target_bone_names)
            if missing:
                raise ValueError(
                    "The target rig is missing vrc->zmd destinations: "
                    + ", ".join(missing)
                )
        mapping = {
            source: target
            for source, target in mapping.items()
            if target in target_bone_names
        }
    return {source: target for source, target in mapping.items() if source in source_names}


def _capture_source_weights(source_obj, mapping):
    """Read sparse RNA weights once; coalesce and normalize with NumPy."""
    target_names = list(dict.fromkeys(mapping.values()))
    target_ids = {name: index for index, name in enumerate(target_names)}
    index_to_target = {
        group.index: target_ids[mapping[group.name]]
        for group in source_obj.vertex_groups if group.name in mapping
    }
    index_to_name = {group.index: group.name for group in source_obj.vertex_groups}
    vertex_count = len(source_obj.data.vertices)
    keys = array("q")
    weights = array("d")
    source_nonempty = set()
    for vertex in source_obj.data.vertices:
        for assignment in vertex.groups:
            target_id = index_to_target.get(assignment.group)
            weight = assignment.weight
            if target_id is None or weight <= _WEIGHT_EPSILON:
                continue
            source_nonempty.add(index_to_name[assignment.group])
            keys.append(target_id * vertex_count + vertex.index)
            weights.append(weight)

    weights_by_target = {}
    normalized_vertex_count = 0
    mapped_vertex_count = 0
    if keys:
        unique_keys, inverse = np.unique(np.asarray(keys), return_inverse=True)
        merged = np.bincount(inverse, weights=np.asarray(weights))
        vertex_ids = unique_keys % vertex_count
        totals = np.bincount(vertex_ids, weights=merged, minlength=vertex_count)
        mapped_vertex_count = int(np.count_nonzero(totals))
        normalized_vertex_count = int(np.count_nonzero(totals > _WEIGHT_EPSILON))
        normalized = merged / totals[vertex_ids]
        keep = normalized > _WEIGHT_EPSILON
        ids = unique_keys[keep] // vertex_count
        vertex_ids = vertex_ids[keep]
        normalized = normalized[keep].astype(np.float32)
        targets, starts, counts = np.unique(ids, return_index=True, return_counts=True)
        for target_id, start, count in zip(targets, starts, counts):
            end = start + count
            weights_by_target[target_names[target_id]] = (
                vertex_ids[start:end], normalized[start:end]
            )

    return {
        "weights_by_target": weights_by_target,
        "source_nonempty": source_nonempty,
        "mapped_vertex_count": mapped_vertex_count,
        "normalized_vertex_count": normalized_vertex_count,
    }


def _replace_source_groups(source_obj, weights_by_target, target_order):
    for group in reversed(list(source_obj.vertex_groups)):
        source_obj.vertex_groups.remove(group)

    created_count = 0
    for target_name in target_order:
        data = weights_by_target.get(target_name)
        if data is None:
            continue
        vertex_ids, weights = data
        group = source_obj.vertex_groups.new(name=target_name)
        # Batch exactly equal float32 weights (especially rigid weight 1.0).
        order = np.argsort(weights, kind="stable")
        sorted_weights = weights[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_weights)) + 1, len(order)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group.add(vertex_ids[order[start:end]].tolist(), float(sorted_weights[start]), "REPLACE")
        created_count += 1
    return created_count


def _apply_group_names_only(source_obj, mapping):
    """Reuse exact renaming without scanning, normalizing or merging weights."""
    conflicts = sorted(target for target, count in Counter(mapping.values()).items() if count > 1)
    if conflicts:
        raise ValueError(
            "Rename destinations already exist; weights were not merged: "
            + ", ".join(conflicts)
        )
    result = apply_exact_name_pairs(source_obj, mapping.items())
    return {
        "created_count": result["renamed_count"],
        "mapped_source_count": len(mapping),
        "removed_group_count": 0,
        "merge_target_count": 0,
        "mapped_vertex_count": 0,
        "normalized_vertex_count": 0,
    }


def apply_auto_rig_npc(
    source_obj,
    mapping_id=AUTO_RIG_NPC_MAPPING_ID,
    *,
    target_bone_names=None,
):
    """Apply a selected game-rig rename preset to one mesh object.

    The preset owns the source-to-target naming contract. This operation only
    changes vertex groups and their sparse weights; it never mutates armatures,
    parents, or modifiers.
    """
    if not is_mesh_object(source_obj):
        raise ValueError("Choose a valid mesh object.")
    if not source_obj.vertex_groups:
        raise ValueError("The active mesh has no vertex groups.")
    mapping_spec = _RIG_MAPPING_SPECS.get(mapping_id)
    if mapping_spec is None:
        raise ValueError(f"Unknown mapping preset: {mapping_id}")

    source_names = {group.name for group in source_obj.vertex_groups}
    mapping = _build_mapping(
        mapping_id,
        source_names,
        target_bone_names=target_bone_names,
    )
    if not mapping:
        raise ValueError(
            f"No source vertex groups matched the '{mapping_spec['label']}' mapping."
        )
    if mapping_id == VRC_ZMD_MAPPING_ID:
        return _apply_group_names_only(source_obj, mapping)
    capture = _capture_source_weights(source_obj, mapping)
    target_order = list(capture["weights_by_target"])

    created_count = _replace_source_groups(
        source_obj,
        capture["weights_by_target"],
        target_order,
    )
    target_counts = Counter(mapping.values())
    return {
        "created_count": created_count,
        "mapped_source_count": len(mapping),
        "removed_group_count": len(source_names) - len(mapping),
        "merge_target_count": sum(count > 1 for count in target_counts.values()),
        "mapped_vertex_count": capture["mapped_vertex_count"],
        "normalized_vertex_count": capture["normalized_vertex_count"],
    }


def apply_auto_rig_npc_batch(
    source_objects,
    mapping_id=AUTO_RIG_NPC_MAPPING_ID,
    *,
    target_bone_names=None,
):
    """Apply a naming preset to every selected mesh with groups."""
    results = []
    for source_obj in source_objects:
        if not is_mesh_object(source_obj) or not source_obj.vertex_groups:
            continue
        result = apply_auto_rig_npc(
            source_obj,
            mapping_id,
            target_bone_names=target_bone_names,
        )
        results.append((source_obj, result))
    return results
