"""Smart X-axis pose mirroring for active pose armatures."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector

from .common import is_armature_object


_PAIR_CACHE = {}
_LAST_SIGNATURES = {}
_MIRROR_REENTRY = False
_POLL_INTERVAL_SECONDS = 0.05
_IDLE_INTERVAL_SECONDS = 0.2
_CENTER_TOLERANCE = 1.0e-4
_MIRROR_X_MATRIX = Matrix(
    (
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def _scene_key(scene):
    return scene.as_pointer()


def _mirror_x_vector(vector):
    return Vector((-vector.x, vector.y, vector.z))


def _matrix_signature(matrix):
    return tuple(round(float(value), 6) for row in matrix for value in row)


def _armature_rest_signature(armature_obj):
    return tuple(
        (data_bone.name, _matrix_signature(data_bone.matrix_local))
        for data_bone in armature_obj.data.bones
    )


def _bone_center_x(data_bone):
    return (float(data_bone.head_local.x) + float(data_bone.tail_local.x)) * 0.5


def _side_sign(value):
    if abs(value) <= _CENTER_TOLERANCE:
        return 0
    return 1 if value > 0.0 else -1


def _bone_info_map(armature_obj):
    info_map = {}
    for data_bone in armature_obj.data.bones:
        axis = data_bone.tail_local - data_bone.head_local
        if axis.length == 0.0:
            axis = Vector((0.0, 0.0, 1.0))
        else:
            axis.normalize()
        info_map[data_bone.name] = {
            "head": data_bone.head_local.copy(),
            "tail": data_bone.tail_local.copy(),
            "axis": axis,
            "length": max(float(data_bone.length), 1.0e-6),
            "center_x": _bone_center_x(data_bone),
        }
    return info_map


def _candidate_score(source_info, candidate_info):
    mirrored_head = _mirror_x_vector(source_info["head"])
    mirrored_tail = _mirror_x_vector(source_info["tail"])
    mirrored_axis = _mirror_x_vector(source_info["axis"])

    head_distance = (mirrored_head - candidate_info["head"]).length
    tail_distance = (mirrored_tail - candidate_info["tail"]).length
    axis_distance = (mirrored_axis - candidate_info["axis"]).length
    length_distance = abs(source_info["length"] - candidate_info["length"])
    return head_distance + tail_distance + (axis_distance * source_info["length"] * 0.35) + (length_distance * 0.2)


def _build_pair_map(armature_obj):
    info_map = _bone_info_map(armature_obj)
    positive_side_names = []
    negative_side_names = []
    for bone_name, bone_info in info_map.items():
        side = _side_sign(bone_info["center_x"])
        if side > 0:
            positive_side_names.append(bone_name)
        elif side < 0:
            negative_side_names.append(bone_name)

    candidate_pairs = []
    for positive_name in positive_side_names:
        positive_info = info_map[positive_name]
        for negative_name in negative_side_names:
            negative_info = info_map[negative_name]
            score = _candidate_score(positive_info, negative_info)
            max_length = max(positive_info["length"], negative_info["length"], 1.0e-6)
            score_limit = max(max_length * 2.5, 0.05)
            if score <= score_limit:
                candidate_pairs.append((score, positive_name, negative_name))

    candidate_pairs.sort(key=lambda item: item[0])
    pair_map = {}
    used_names = set()
    for _score, positive_name, negative_name in candidate_pairs:
        if positive_name in used_names or negative_name in used_names:
            continue
        pair_map[positive_name] = negative_name
        pair_map[negative_name] = positive_name
        used_names.add(positive_name)
        used_names.add(negative_name)

    return pair_map, info_map


def _pose_mirror_pair_map(armature_obj):
    cache_key = armature_obj.data.as_pointer()
    rest_signature = _armature_rest_signature(armature_obj)
    cached_entry = _PAIR_CACHE.get(cache_key)
    if cached_entry is not None and cached_entry["signature"] == rest_signature:
        return cached_entry["pair_map"], cached_entry["info_map"]

    pair_map, info_map = _build_pair_map(armature_obj)
    _PAIR_CACHE[cache_key] = {
        "signature": rest_signature,
        "pair_map": pair_map,
        "info_map": info_map,
    }
    return pair_map, info_map


def _active_pose_armature(context):
    armature_obj = getattr(context, "active_object", None)
    if not is_armature_object(armature_obj):
        return None
    if getattr(context, "mode", "") != "POSE":
        return None
    return armature_obj


def _selected_driver_names(context, armature_obj, pair_map, info_map):
    selected_pose_bones = list(getattr(context, "selected_pose_bones_from_active_object", None) or [])
    selected_names = [pose_bone.name for pose_bone in selected_pose_bones if pose_bone.name in pair_map]

    active_pose_bone = getattr(context, "active_pose_bone", None)
    active_name = (
        active_pose_bone.name
        if active_pose_bone is not None and getattr(active_pose_bone, "id_data", None) == armature_obj
        else ""
    )

    if active_name and active_name in pair_map and active_name not in selected_names:
        selected_names.insert(0, active_name)
    if not selected_names:
        return ()

    selected_name_set = set(selected_names)
    driver_names = []
    processed_names = set()

    for selected_name in selected_names:
        if selected_name in processed_names:
            continue
        partner_name = pair_map.get(selected_name, "")
        if not partner_name:
            continue

        if partner_name in selected_name_set:
            if active_name in {selected_name, partner_name}:
                driver_name = active_name
            else:
                selected_center_x = info_map[selected_name]["center_x"]
                partner_center_x = info_map[partner_name]["center_x"]
                driver_name = selected_name if selected_center_x >= partner_center_x else partner_name
            processed_names.add(selected_name)
            processed_names.add(partner_name)
            if driver_name and driver_name not in driver_names:
                driver_names.append(driver_name)
            continue

        processed_names.add(selected_name)
        driver_names.append(selected_name)

    return tuple(driver_names)


def _smart_pose_mirror_enabled(scene):
    settings = getattr(scene, "mdt_pose_mirror_settings", None) if scene is not None else None
    return bool(settings is not None and getattr(settings, "enabled", False))


def _smart_mirror_signature(context, scene):
    if not _smart_pose_mirror_enabled(scene):
        return ()

    armature_obj = _active_pose_armature(context)
    if armature_obj is None:
        return ()

    pair_map, info_map = _pose_mirror_pair_map(armature_obj)
    driver_names = _selected_driver_names(context, armature_obj, pair_map, info_map)
    if not driver_names:
        return ()

    signature = [armature_obj.as_pointer(), tuple(driver_names)]
    for bone_name in driver_names:
        pose_bone = armature_obj.pose.bones.get(bone_name)
        if pose_bone is not None:
            signature.append((bone_name, _matrix_signature(pose_bone.matrix)))
    return tuple(signature)


def _apply_smart_x_mirror(context):
    armature_obj = _active_pose_armature(context)
    if armature_obj is None:
        return 0

    pair_map, info_map = _pose_mirror_pair_map(armature_obj)
    driver_names = _selected_driver_names(context, armature_obj, pair_map, info_map)
    if not driver_names:
        return 0

    mirrored_count = 0
    for driver_name in driver_names:
        partner_name = pair_map.get(driver_name, "")
        if not partner_name:
            continue

        driver_pose_bone = armature_obj.pose.bones.get(driver_name)
        partner_pose_bone = armature_obj.pose.bones.get(partner_name)
        driver_data_bone = armature_obj.data.bones.get(driver_name)
        partner_data_bone = armature_obj.data.bones.get(partner_name)
        if (
            driver_pose_bone is None
            or partner_pose_bone is None
            or driver_data_bone is None
            or partner_data_bone is None
        ):
            continue

        source_delta = driver_pose_bone.matrix.copy() @ driver_data_bone.matrix_local.inverted()
        mirrored_delta = _MIRROR_X_MATRIX @ source_delta @ _MIRROR_X_MATRIX
        partner_pose_bone.matrix = mirrored_delta @ partner_data_bone.matrix_local
        mirrored_count += 1

    if mirrored_count > 0:
        context.view_layer.update()
    return mirrored_count


def _run_smart_x_mirror(context, scene):
    global _MIRROR_REENTRY

    if _MIRROR_REENTRY or not _smart_pose_mirror_enabled(scene):
        return

    scene_key = _scene_key(scene)
    current_signature = _smart_mirror_signature(context, scene)
    previous_signature = _LAST_SIGNATURES.get(scene_key)
    if current_signature == previous_signature:
        return

    _LAST_SIGNATURES[scene_key] = current_signature
    if not current_signature:
        return

    _MIRROR_REENTRY = True
    try:
        _apply_smart_x_mirror(context)
    finally:
        _LAST_SIGNATURES[scene_key] = _smart_mirror_signature(context, scene)
        _MIRROR_REENTRY = False


@persistent
def _pose_mirror_depsgraph_update(scene, _depsgraph):
    _run_smart_x_mirror(bpy.context, scene)


def _pose_mirror_timer():
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None:
        return _IDLE_INTERVAL_SECONDS
    if not _smart_pose_mirror_enabled(scene):
        return _IDLE_INTERVAL_SECONDS
    _run_smart_x_mirror(context, scene)
    return _POLL_INTERVAL_SECONDS


def clear_pose_mirror_state():
    """Drop cached pairs and runtime pose signatures."""
    _PAIR_CACHE.clear()
    _LAST_SIGNATURES.clear()


def register_pose_mirror_handler():
    """Register smart pose mirror handlers once."""
    depsgraph_handlers = bpy.app.handlers.depsgraph_update_post
    if _pose_mirror_depsgraph_update not in depsgraph_handlers:
        depsgraph_handlers.append(_pose_mirror_depsgraph_update)
    if not bpy.app.timers.is_registered(_pose_mirror_timer):
        bpy.app.timers.register(_pose_mirror_timer, first_interval=_POLL_INTERVAL_SECONDS, persistent=True)


def unregister_pose_mirror_handler():
    """Remove smart pose mirror handlers and cached state."""
    depsgraph_handlers = bpy.app.handlers.depsgraph_update_post
    if _pose_mirror_depsgraph_update in depsgraph_handlers:
        depsgraph_handlers.remove(_pose_mirror_depsgraph_update)
    if bpy.app.timers.is_registered(_pose_mirror_timer):
        bpy.app.timers.unregister(_pose_mirror_timer)
    clear_pose_mirror_state()
