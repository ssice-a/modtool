"""Armature cleanup helpers for Mod Tool."""

from __future__ import annotations

import bpy

from .common import get_selected_mesh_objects, is_mesh_object


_WEIGHT_EPSILON = 1.0e-8
_POSE_VECTOR_LOCK_ATTRIBUTES = ("lock_location", "lock_rotation", "lock_scale")
_POSE_BOOL_LOCK_ATTRIBUTES = (
    "lock_rotation_w",
    "lock_rotations_4d",
    "lock_ik_x",
    "lock_ik_y",
    "lock_ik_z",
)
_POSE_IK_LIMIT_ATTRIBUTES = (
    "use_ik_limit_x",
    "use_ik_limit_y",
    "use_ik_limit_z",
    "use_ik_linear_control",
    "use_ik_rotation_control",
)
_BONE_ACCESS_LOCK_ATTRIBUTES = ("hide", "hide_select")


def is_armature_object(obj):
    """Return True when the object is an armature."""
    return obj is not None and getattr(obj, "type", None) == "ARMATURE"


def mesh_uses_armature(obj, armature_obj):
    """Return True when a mesh has an Armature modifier targeting the armature."""
    if not is_mesh_object(obj) or not is_armature_object(armature_obj):
        return False

    return any(
        modifier.type == "ARMATURE" and modifier.object == armature_obj
        for modifier in getattr(obj, "modifiers", [])
    )


def _append_unique(items, item):
    if item is not None and item not in items:
        items.append(item)


def _armatures_from_meshes(mesh_objects):
    armatures = []
    for obj in mesh_objects:
        for modifier in getattr(obj, "modifiers", []):
            if modifier.type != "ARMATURE":
                continue
            _append_unique(armatures, modifier.object)
    return [obj for obj in armatures if is_armature_object(obj)]


def resolve_cleanup_armature(context):
    """Resolve the armature that should be cleaned from the current selection."""
    active_obj = getattr(context, "object", None)
    if is_armature_object(active_obj):
        return active_obj, ""

    selected_objects = list(getattr(context, "selected_objects", []))
    selected_armatures = [obj for obj in selected_objects if is_armature_object(obj)]
    if len(selected_armatures) == 1:
        return selected_armatures[0], ""
    if len(selected_armatures) > 1:
        return None, "Select only one armature to clean."

    selected_meshes = get_selected_mesh_objects(context)
    mesh_armatures = _armatures_from_meshes(selected_meshes)
    if len(mesh_armatures) == 1:
        return mesh_armatures[0], ""
    if len(mesh_armatures) > 1:
        return None, "Selected meshes use more than one armature."

    return None, "Select an armature, or select a mesh that uses one Armature modifier."


def get_cleanup_mesh_objects(context, armature_obj):
    """Return meshes whose weights should decide which bones are kept."""
    selected_meshes = get_selected_mesh_objects(context)
    selected_linked_meshes = [
        obj for obj in selected_meshes if mesh_uses_armature(obj, armature_obj)
    ]
    if selected_linked_meshes:
        return selected_linked_meshes

    scene_linked_meshes = [
        obj
        for obj in getattr(context.scene, "objects", [])
        if mesh_uses_armature(obj, armature_obj)
    ]
    if scene_linked_meshes:
        return scene_linked_meshes

    return selected_meshes


def get_weighted_vertex_group_names(mesh_objects, weight_epsilon=_WEIGHT_EPSILON):
    """Collect vertex-group names that have at least one effective weight."""
    weighted_names = set()
    for obj in mesh_objects:
        group_names_by_index = {group.index: group.name for group in obj.vertex_groups}
        for vertex in getattr(obj.data, "vertices", []):
            for assignment in vertex.groups:
                if assignment.weight <= weight_epsilon:
                    continue
                group_name = group_names_by_index.get(assignment.group)
                if group_name:
                    weighted_names.add(group_name)
    return weighted_names


def count_pose_constraints(armature_obj):
    """Count all pose-bone constraints on an armature."""
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return 0
    return sum(len(pose_bone.constraints) for pose_bone in armature_obj.pose.bones)


def _is_target_pose_bone(pose_bone, bone_names):
    return bone_names is None or pose_bone.name in bone_names


def _count_locked_vector_channels(pose_bone, attribute_name):
    return sum(1 for is_locked in getattr(pose_bone, attribute_name, ()) if is_locked)


def count_pose_locks(armature_obj, bone_names=None):
    """Count transform, IK, and selection locks on pose bones."""
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return 0

    target_bone_names = set(bone_names) if bone_names is not None else None
    lock_count = 0
    for pose_bone in armature_obj.pose.bones:
        if not _is_target_pose_bone(pose_bone, target_bone_names):
            continue

        for attribute_name in _POSE_VECTOR_LOCK_ATTRIBUTES:
            lock_count += _count_locked_vector_channels(pose_bone, attribute_name)
        for attribute_name in _POSE_BOOL_LOCK_ATTRIBUTES + _POSE_IK_LIMIT_ATTRIBUTES:
            if getattr(pose_bone, attribute_name, False):
                lock_count += 1
        for attribute_name in _BONE_ACCESS_LOCK_ATTRIBUTES:
            if getattr(pose_bone.bone, attribute_name, False):
                lock_count += 1
    return lock_count


def remove_pose_constraints(armature_obj):
    """Remove every pose-bone constraint from an armature."""
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return 0

    removed_count = 0
    for pose_bone in armature_obj.pose.bones:
        for constraint in reversed(list(pose_bone.constraints)):
            pose_bone.constraints.remove(constraint)
            removed_count += 1
    return removed_count


def clear_pose_locks(armature_obj, bone_names=None):
    """Clear transform, IK, and selection locks from pose bones."""
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return 0

    target_bone_names = set(bone_names) if bone_names is not None else None
    cleared_count = count_pose_locks(armature_obj, target_bone_names)
    for pose_bone in armature_obj.pose.bones:
        if not _is_target_pose_bone(pose_bone, target_bone_names):
            continue

        pose_bone.lock_location = (False, False, False)
        pose_bone.lock_rotation = (False, False, False)
        pose_bone.lock_scale = (False, False, False)
        for attribute_name in _POSE_BOOL_LOCK_ATTRIBUTES + _POSE_IK_LIMIT_ATTRIBUTES:
            if hasattr(pose_bone, attribute_name):
                setattr(pose_bone, attribute_name, False)
        for attribute_name in _BONE_ACCESS_LOCK_ATTRIBUTES:
            if hasattr(pose_bone.bone, attribute_name):
                setattr(pose_bone.bone, attribute_name, False)
    return cleared_count


def _build_nearest_kept_parent_map(parent_map, keep_bone_names):
    assignments = {}
    for bone_name in keep_bone_names:
        parent_name = parent_map.get(bone_name)
        while parent_name is not None and parent_name not in keep_bone_names:
            parent_name = parent_map.get(parent_name)
        assignments[bone_name] = parent_name
    return assignments


def clean_fk_armature(context, armature_obj, mesh_objects, weight_epsilon=_WEIGHT_EPSILON):
    """Keep weighted bones, preserve retained hierarchy, and strip pose constraints."""
    if not is_armature_object(armature_obj):
        raise ValueError("Choose a valid armature.")
    if not mesh_objects:
        raise ValueError("Select at least one mesh, or use an armature that has linked mesh objects.")

    weighted_group_names = get_weighted_vertex_group_names(mesh_objects, weight_epsilon)
    original_bone_names = {bone.name for bone in armature_obj.data.bones}
    keep_bone_names = weighted_group_names.intersection(original_bone_names)
    if not keep_bone_names:
        raise ValueError("No weighted vertex groups match bones on the armature.")

    parent_map = {
        bone.name: bone.parent.name if bone.parent is not None else None
        for bone in armature_obj.data.bones
    }
    parent_assignments = _build_nearest_kept_parent_map(parent_map, keep_bone_names)
    original_bone_count = len(original_bone_names)

    constraint_count = remove_pose_constraints(armature_obj)
    lock_count = clear_pose_locks(armature_obj, keep_bone_names)

    context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = armature_obj.data.edit_bones
        for bone_name, parent_name in parent_assignments.items():
            edit_bone = edit_bones.get(bone_name)
            if edit_bone is None:
                continue

            parent_bone = edit_bones.get(parent_name) if parent_name else None
            if edit_bone.parent != parent_bone:
                edit_bone.parent = parent_bone
                edit_bone.use_connect = False

        removed_bone_count = 0
        for bone_name in original_bone_names.difference(keep_bone_names):
            edit_bone = edit_bones.get(bone_name)
            if edit_bone is None:
                continue
            edit_bones.remove(edit_bone)
            removed_bone_count += 1
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    return {
        "constraint_count": constraint_count,
        "lock_count": lock_count,
        "kept_bone_count": len(keep_bone_names),
        "removed_bone_count": removed_bone_count,
        "original_bone_count": original_bone_count,
        "mesh_count": len(mesh_objects),
        "weighted_group_count": len(weighted_group_names),
    }
