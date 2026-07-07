"""Armature FK cleanup helpers."""

from __future__ import annotations

from .common import is_armature_object


def resolve_cleanup_armature(context):
    """Return the armature to clean from the active or selected objects."""
    active_obj = getattr(context, "object", None)
    if is_armature_object(active_obj):
        return active_obj, None

    selected_armatures = [
        obj
        for obj in getattr(context, "selected_objects", [])
        if is_armature_object(obj)
    ]
    if len(selected_armatures) == 1:
        return selected_armatures[0], None
    if len(selected_armatures) > 1:
        return None, "Make the armature to clean active, or select only one armature."
    return None, "Select an armature to clean."


def _clear_bool_attribute(owner, attribute_name):
    if not hasattr(owner, attribute_name):
        return 0

    try:
        if not getattr(owner, attribute_name):
            return 0
        setattr(owner, attribute_name, False)
    except (AttributeError, TypeError, ValueError):
        return 0
    return 1


def _clear_bool_vector(owner, attribute_name):
    if not hasattr(owner, attribute_name):
        return 0

    try:
        values = tuple(bool(value) for value in getattr(owner, attribute_name))
    except (AttributeError, TypeError, ValueError):
        return 0

    if not any(values):
        return 0

    try:
        setattr(owner, attribute_name, (False,) * len(values))
    except (AttributeError, TypeError, ValueError):
        return 0
    return sum(values)


def _remove_pose_constraints(armature_obj):
    constraint_count = 0
    for pose_bone in armature_obj.pose.bones:
        constraints = pose_bone.constraints
        constraint_count += len(constraints)
        for constraint in reversed(list(constraints)):
            constraints.remove(constraint)
    return constraint_count


def _clear_pose_locks(armature_obj):
    lock_count = 0
    for pose_bone in armature_obj.pose.bones:
        lock_count += _clear_bool_vector(pose_bone, "lock_location")
        lock_count += _clear_bool_vector(pose_bone, "lock_rotation")
        lock_count += _clear_bool_vector(pose_bone, "lock_scale")
        lock_count += _clear_bool_attribute(pose_bone, "lock_rotation_w")
        lock_count += _clear_bool_attribute(pose_bone, "lock_rotations_4d")

        lock_count += _clear_bool_attribute(pose_bone, "lock_ik_x")
        lock_count += _clear_bool_attribute(pose_bone, "lock_ik_y")
        lock_count += _clear_bool_attribute(pose_bone, "lock_ik_z")
        lock_count += _clear_bool_attribute(pose_bone, "use_ik_limit_x")
        lock_count += _clear_bool_attribute(pose_bone, "use_ik_limit_y")
        lock_count += _clear_bool_attribute(pose_bone, "use_ik_limit_z")
        lock_count += _clear_bool_attribute(pose_bone, "use_ik_rotation_control")
        lock_count += _clear_bool_attribute(pose_bone, "use_ik_linear_control")

        bone = pose_bone.bone
        lock_count += _clear_bool_attribute(bone, "hide_select")
        lock_count += _clear_bool_attribute(bone, "hide")
    return lock_count


def _capture_pose_matrices(armature_obj):
    return {
        pose_bone.name: pose_bone.matrix.copy()
        for pose_bone in armature_obj.pose.bones
    }


def _pose_bones_parent_first(armature_obj):
    depth_by_name = {}

    def bone_depth(bone):
        cached_depth = depth_by_name.get(bone.name)
        if cached_depth is not None:
            return cached_depth
        if bone.parent is None:
            depth = 0
        else:
            depth = bone_depth(bone.parent) + 1
        depth_by_name[bone.name] = depth
        return depth

    return sorted(
        armature_obj.pose.bones,
        key=lambda pose_bone: bone_depth(pose_bone.bone),
    )


def _restore_pose_matrices(armature_obj, pose_matrices):
    restored_count = 0
    for pose_bone in _pose_bones_parent_first(armature_obj):
        pose_matrix = pose_matrices.get(pose_bone.name)
        if pose_matrix is None:
            continue
        pose_bone.matrix = pose_matrix
        restored_count += 1
    return restored_count


def clean_fk_armature(context, armature_obj):
    """Strip pose constraints/locks while preserving the current visual pose."""
    if not is_armature_object(armature_obj):
        raise ValueError("Clean To FK needs an armature object.")

    context.view_layer.update()
    pose_matrices = _capture_pose_matrices(armature_obj)

    constraint_count = _remove_pose_constraints(armature_obj)
    lock_count = _clear_pose_locks(armature_obj)
    context.view_layer.update()

    restored_pose_count = _restore_pose_matrices(armature_obj, pose_matrices)
    context.view_layer.update()

    bone_count = len(armature_obj.data.bones)
    return {
        "bone_count": bone_count,
        "constraint_count": constraint_count,
        "lock_count": lock_count,
        "restored_pose_count": restored_pose_count,
    }
