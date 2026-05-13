"""Armature cleanup helpers for Mod Tool."""

from __future__ import annotations

import bpy

from .common import get_selected_mesh_objects, is_armature_object, is_mesh_object, run_object_operator
from .modifier_apply import apply_modifier_with_shape_keys, get_enabled_modifier_names, has_shape_keys


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
    cleared_count = 0
    for pose_bone in armature_obj.pose.bones:
        if not _is_target_pose_bone(pose_bone, target_bone_names):
            continue

        for attribute_name in _POSE_VECTOR_LOCK_ATTRIBUTES:
            cleared_count += sum(1 for is_locked in getattr(pose_bone, attribute_name, ()) if is_locked)
        pose_bone.lock_location = (False, False, False)
        pose_bone.lock_rotation = (False, False, False)
        pose_bone.lock_scale = (False, False, False)
        for attribute_name in _POSE_BOOL_LOCK_ATTRIBUTES + _POSE_IK_LIMIT_ATTRIBUTES:
            if getattr(pose_bone, attribute_name, False):
                cleared_count += 1
            if hasattr(pose_bone, attribute_name):
                setattr(pose_bone, attribute_name, False)
        for attribute_name in _BONE_ACCESS_LOCK_ATTRIBUTES:
            if getattr(pose_bone.bone, attribute_name, False):
                cleared_count += 1
            if hasattr(pose_bone.bone, attribute_name):
                setattr(pose_bone.bone, attribute_name, False)
    return cleared_count


def clear_pose_transforms(armature_obj, bone_names=None):
    """Clear pose transforms after the current pose has been applied as rest pose."""
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return 0

    target_bone_names = set(bone_names) if bone_names is not None else None
    cleared_count = 0
    for pose_bone in armature_obj.pose.bones:
        if not _is_target_pose_bone(pose_bone, target_bone_names):
            continue

        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
        if pose_bone.rotation_mode == "QUATERNION":
            pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        elif pose_bone.rotation_mode == "AXIS_ANGLE":
            pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        else:
            pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        cleared_count += 1
    return cleared_count


def _build_nearest_kept_parent_map(parent_map, keep_bone_names):
    assignments = {}
    for bone_name in keep_bone_names:
        parent_name = parent_map.get(bone_name)
        while parent_name is not None and parent_name not in keep_bone_names:
            parent_name = parent_map.get(parent_name)
        assignments[bone_name] = parent_name
    return assignments


def _capture_armature_modifier_settings(modifier):
    settings = {
        "name": modifier.name,
        "show_viewport": modifier.show_viewport,
        "show_render": modifier.show_render,
        "show_in_editmode": getattr(modifier, "show_in_editmode", False),
        "show_on_cage": getattr(modifier, "show_on_cage", False),
        "show_expanded": getattr(modifier, "show_expanded", True),
    }
    for attribute_name in (
        "use_vertex_groups",
        "use_bone_envelopes",
        "use_deform_preserve_volume",
        "use_multi_modifier",
        "vertex_group",
        "invert_vertex_group",
    ):
        if hasattr(modifier, attribute_name):
            settings[attribute_name] = getattr(modifier, attribute_name)
    return settings


def _capture_armature_modifier_bindings(mesh_objects, armature_obj):
    bindings = {}
    for obj in mesh_objects:
        object_bindings = []
        for modifier in obj.modifiers:
            if modifier.type != "ARMATURE" or modifier.object != armature_obj:
                continue
            if not modifier.show_viewport:
                continue
            object_bindings.append(_capture_armature_modifier_settings(modifier))
        if object_bindings:
            bindings[obj.name] = object_bindings
    return bindings


def _apply_modifier_settings(modifier, settings):
    for attribute_name, value in settings.items():
        if attribute_name == "name" or not hasattr(modifier, attribute_name):
            continue
        try:
            setattr(modifier, attribute_name, value)
        except (AttributeError, TypeError):
            pass


def _restore_armature_modifier_bindings(mesh_objects, armature_obj, bindings):
    restored_count = 0
    for obj in mesh_objects:
        for settings in bindings.get(obj.name, []):
            modifier = obj.modifiers.new(settings["name"], "ARMATURE")
            modifier.object = armature_obj
            _apply_modifier_settings(modifier, settings)
            modifier.show_viewport = True
            restored_count += 1
    return restored_count


def _remove_armature_modifiers(mesh_objects, armature_obj):
    removed_count = 0
    for obj in mesh_objects:
        for modifier in reversed(list(obj.modifiers)):
            if modifier.type != "ARMATURE" or modifier.object != armature_obj:
                continue
            obj.modifiers.remove(modifier)
            removed_count += 1
    return removed_count


def _apply_enabled_modifiers_without_shape_keys(context, obj):
    applied_names = []
    for modifier_name in get_enabled_modifier_names(obj):
        modifier = obj.modifiers.get(modifier_name)
        if modifier is None or not modifier.show_viewport:
            continue

        try:
            result = run_object_operator(context, obj, "modifier_apply", modifier=modifier.name)
        except RuntimeError as exc:
            raise ValueError(f"Could not apply modifier '{modifier_name}' on {obj.name}: {exc}") from exc
        if "FINISHED" not in result:
            raise ValueError(f"Could not apply modifier '{modifier_name}' on {obj.name}.")
        applied_names.append(modifier_name)
    return {
        "applied_modifier_names": applied_names,
        "shape_key_count": 0,
        "vertex_count": len(obj.data.vertices),
        "vertex_group_count": len(obj.vertex_groups),
    }


def _apply_mesh_modifiers_preserving_shape_keys(context, mesh_objects):
    results = []
    for obj in mesh_objects:
        if len(get_enabled_modifier_names(obj)) == 0:
            results.append((obj.name, {"applied_modifier_names": [], "shape_key_count": 0}))
            continue

        if has_shape_keys(obj):
            results.append((obj.name, apply_modifier_with_shape_keys(context, obj)))
        else:
            results.append((obj.name, _apply_enabled_modifiers_without_shape_keys(context, obj)))
    return results


def _apply_current_pose_as_rest(context, armature_obj):
    if not is_armature_object(armature_obj) or armature_obj.pose is None:
        return False

    context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    try:
        try:
            result = bpy.ops.pose.armature_apply(selected=False)
        except TypeError:
            result = bpy.ops.pose.armature_apply()
        if "FINISHED" not in result:
            raise ValueError("Could not apply the current armature pose as the new rest pose.")
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    context.view_layer.update()
    return True


def _remove_unweighted_bones(armature_obj, keep_bone_names, parent_assignments):
    original_bone_names = {bone.name for bone in armature_obj.data.bones}
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

    return removed_bone_count


def clean_fk_armature(context, armature_obj, mesh_objects, weight_epsilon=_WEIGHT_EPSILON):
    """Bake visible mesh modifiers, turn the current armature pose into rest pose, and rebuild FK bindings."""
    if not is_armature_object(armature_obj):
        raise ValueError("Choose a valid armature.")
    if not mesh_objects:
        raise ValueError("Select at least one mesh, or use an armature that has linked mesh objects.")

    mesh_objects = [obj for obj in mesh_objects if is_mesh_object(obj)]
    if not mesh_objects:
        raise ValueError("Select at least one mesh object.")

    original_bone_names = {bone.name for bone in armature_obj.data.bones}
    original_bone_count = len(original_bone_names)
    armature_bindings = _capture_armature_modifier_bindings(mesh_objects, armature_obj)
    if not armature_bindings:
        raise ValueError("No viewport-enabled Armature modifier using this armature was found.")

    modifier_results = _apply_mesh_modifiers_preserving_shape_keys(context, mesh_objects)
    removed_leftover_armature_modifier_count = _remove_armature_modifiers(mesh_objects, armature_obj)

    weighted_group_names = get_weighted_vertex_group_names(mesh_objects, weight_epsilon)
    keep_bone_names = weighted_group_names.intersection(original_bone_names)
    if not keep_bone_names:
        raise ValueError("No weighted vertex groups match bones on the armature after applying modifiers.")

    parent_map = {
        bone.name: bone.parent.name if bone.parent is not None else None
        for bone in armature_obj.data.bones
    }
    parent_assignments = _build_nearest_kept_parent_map(parent_map, keep_bone_names)

    _apply_current_pose_as_rest(context, armature_obj)
    constraint_count = remove_pose_constraints(armature_obj)
    context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    transform_count = clear_pose_transforms(armature_obj)
    lock_count = clear_pose_locks(armature_obj)
    removed_bone_count = _remove_unweighted_bones(armature_obj, keep_bone_names, parent_assignments)
    restored_modifier_count = _restore_armature_modifier_bindings(mesh_objects, armature_obj, armature_bindings)
    context.view_layer.update()

    return {
        "applied_modifier_count": sum(len(result["applied_modifier_names"]) for _obj_name, result in modifier_results),
        "shape_key_count": sum(result.get("shape_key_count", 0) for _obj_name, result in modifier_results),
        "constraint_count": constraint_count,
        "transform_count": transform_count,
        "lock_count": lock_count,
        "kept_bone_count": len(keep_bone_names),
        "removed_bone_count": removed_bone_count,
        "restored_modifier_count": restored_modifier_count,
        "removed_leftover_armature_modifier_count": removed_leftover_armature_modifier_count,
        "original_bone_count": original_bone_count,
        "mesh_count": len(mesh_objects),
        "weighted_group_count": len(weighted_group_names),
    }
