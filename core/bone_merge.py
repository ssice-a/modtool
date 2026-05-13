"""Armature bone merge helpers for MMD body transfer workflows."""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from .common import is_armature_object


_MIN_BONE_LENGTH = 1.0e-6


@dataclass(frozen=True)
class BoneTransferSpec:
    """Rest-pose data copied from one source bone."""

    name: str
    parent_name: str
    head_world: object
    tail_world: object
    roll: float
    use_deform: bool
    use_connect: bool


def serialize_bone_names(bone_names):
    """Serialize ordered bone names into a compact scene property string."""
    return "\n".join(name for name in bone_names if name)


def parse_bone_names(serialized_names):
    """Parse ordered bone names from a scene property string."""
    seen_names = set()
    names = []
    for raw_name in serialized_names.splitlines():
        name = raw_name.strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        names.append(name)
    return names


def resolve_bone_merge_armatures(context, settings):
    """Resolve source and target armatures from settings with light active-selection fallback."""
    source_obj = getattr(settings, "source_object", None)
    target_obj = getattr(settings, "target_object", None)
    active_obj = getattr(context, "object", None)

    if not is_armature_object(source_obj) and is_armature_object(active_obj):
        source_obj = active_obj

    if not is_armature_object(target_obj):
        selected_armatures = [
            obj
            for obj in getattr(context, "selected_objects", [])
            if is_armature_object(obj) and obj != source_obj
        ]
        if len(selected_armatures) == 1:
            target_obj = selected_armatures[0]

    if not is_armature_object(source_obj):
        source_obj = None
    if not is_armature_object(target_obj):
        target_obj = None
    if source_obj is not None and target_obj is not None and source_obj == target_obj:
        return source_obj, None
    return source_obj, target_obj


def _active_source_selected_pose_bone_names(context, source_obj):
    if getattr(context, "object", None) != source_obj:
        return []

    selected_pose_bones = getattr(context, "selected_pose_bones", None)
    if not selected_pose_bones:
        selected_pose_bones = getattr(context, "selected_pose_bones_from_active_object", None)
    if not selected_pose_bones:
        return []

    return [
        pose_bone.name
        for pose_bone in selected_pose_bones
        if getattr(pose_bone, "id_data", None) in (source_obj, source_obj.data)
    ]


def _active_source_selected_edit_bone_names(context, source_obj):
    if getattr(context, "object", None) != source_obj:
        return []
    if getattr(source_obj, "mode", "") != "EDIT":
        return []

    selected_names = []
    for edit_bone in source_obj.data.edit_bones:
        if (
            getattr(edit_bone, "select", False)
            or getattr(edit_bone, "select_head", False)
            or getattr(edit_bone, "select_tail", False)
        ):
            selected_names.append(edit_bone.name)
    return selected_names


def collect_selected_source_bone_names(context, source_obj):
    """Collect selected bone names from the active source armature."""
    if not is_armature_object(source_obj):
        raise ValueError("Choose a valid source armature.")

    names = _active_source_selected_edit_bone_names(context, source_obj)
    if not names:
        names = _active_source_selected_pose_bone_names(context, source_obj)
    if not names:
        raise ValueError("Select bones on the source armature before capturing.")

    source_order = {bone.name: index for index, bone in enumerate(source_obj.data.bones)}
    names = sorted(set(names), key=lambda name: source_order.get(name, len(source_order)))
    return names


def _ordered_existing_bone_names(source_obj, requested_names):
    requested_set = set(requested_names)
    ordered_names = [bone.name for bone in source_obj.data.bones if bone.name in requested_set]
    missing_names = [name for name in requested_names if source_obj.data.bones.get(name) is None]
    return ordered_names, missing_names


def _bone_world_point(armature_obj, local_point):
    return armature_obj.matrix_world @ local_point


def _collect_source_specs(source_obj, bone_names):
    ordered_names, missing_names = _ordered_existing_bone_names(source_obj, bone_names)
    specs = []
    for bone_name in ordered_names:
        bone = source_obj.data.bones.get(bone_name)
        if bone is None:
            continue
        parent = bone.parent
        specs.append(
            BoneTransferSpec(
                name=bone.name,
                parent_name=parent.name if parent is not None else "",
                head_world=_bone_world_point(source_obj, bone.head_local),
                tail_world=_bone_world_point(source_obj, bone.tail_local),
                roll=float(getattr(bone, "roll", 0.0)),
                use_deform=bool(getattr(bone, "use_deform", True)),
                use_connect=bool(getattr(bone, "use_connect", False)),
            )
        )
    return specs, missing_names


def _nearest_available_parent_name(source_obj, source_bone_name, target_bone_names):
    bone = source_obj.data.bones.get(source_bone_name)
    parent = bone.parent if bone is not None else None
    while parent is not None:
        if parent.name in target_bone_names:
            return parent.name
        parent = parent.parent
    return ""


def _source_bone_has_parent(source_obj, bone_name):
    bone = source_obj.data.bones.get(bone_name)
    return bone is not None and bone.parent is not None


def _count_constraint_refs_to_bones(armature_obj, bone_names):
    if armature_obj.pose is None:
        return {
            "constraints_on_updated_count": 0,
            "constraints_targeting_updated_count": 0,
            "affected_ik_count": 0,
            "invalid_subtarget_count": 0,
        }

    target_names = set(bone_names)
    constraints_on_updated_count = 0
    constraints_targeting_updated_count = 0
    affected_ik_count = 0
    invalid_subtarget_count = 0

    for pose_bone in armature_obj.pose.bones:
        if pose_bone.name in target_names:
            constraints_on_updated_count += len(pose_bone.constraints)

        for constraint in pose_bone.constraints:
            target_obj = getattr(constraint, "target", None)
            subtarget = getattr(constraint, "subtarget", "")
            if target_obj is not None and getattr(target_obj, "type", None) == "ARMATURE" and subtarget:
                if subtarget not in target_obj.data.bones:
                    invalid_subtarget_count += 1
                if target_obj == armature_obj and subtarget in target_names:
                    constraints_targeting_updated_count += 1

            if constraint.type == "IK" and _ik_chain_touches_bones(pose_bone, constraint, target_names):
                affected_ik_count += 1

    return {
        "constraints_on_updated_count": constraints_on_updated_count,
        "constraints_targeting_updated_count": constraints_targeting_updated_count,
        "affected_ik_count": affected_ik_count,
        "invalid_subtarget_count": invalid_subtarget_count,
    }


def _ik_chain_touches_bones(pose_bone, constraint, bone_names):
    chain_count = int(getattr(constraint, "chain_count", 0))
    current_bone = pose_bone
    visited_count = 0
    while current_bone is not None:
        if current_bone.name in bone_names:
            return True
        visited_count += 1
        if chain_count > 0 and visited_count >= chain_count:
            break
        current_bone = current_bone.parent
    return False


def analyze_bone_merge(source_obj, target_obj, bone_names):
    """Analyze how selected source bones will be applied to the target armature."""
    if not is_armature_object(source_obj):
        raise ValueError("Choose a valid source armature.")
    if not is_armature_object(target_obj):
        raise ValueError("Choose a valid target armature.")
    if source_obj == target_obj:
        raise ValueError("Source and Target armatures must be different.")

    requested_names = list(bone_names)
    if not requested_names:
        raise ValueError("Capture source bones before applying.")

    ordered_names, missing_source_names = _ordered_existing_bone_names(source_obj, requested_names)
    if not ordered_names:
        raise ValueError("None of the captured bones exist on the source armature.")

    target_bone_names = {bone.name for bone in target_obj.data.bones}
    updated_names = [name for name in ordered_names if name in target_bone_names]
    added_names = [name for name in ordered_names if name not in target_bone_names]
    available_after_apply = target_bone_names.union(ordered_names)
    missing_parent_names = [
        name
        for name in ordered_names
        if _nearest_available_parent_name(source_obj, name, available_after_apply) == ""
        and _source_bone_has_parent(source_obj, name)
    ]
    constraint_counts = _count_constraint_refs_to_bones(target_obj, updated_names)

    return {
        "requested_count": len(requested_names),
        "source_count": len(ordered_names),
        "updated_count": len(updated_names),
        "added_count": len(added_names),
        "missing_source_count": len(missing_source_names),
        "missing_parent_count": len(missing_parent_names),
        **constraint_counts,
    }


def _save_active_selection(context):
    return context.view_layer.objects.active, list(getattr(context, "selected_objects", []))


def _activate_only(context, obj):
    for selected_obj in list(getattr(context, "selected_objects", [])):
        selected_obj.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def _restore_active_selection(context, previous_active, previous_selected):
    for selected_obj in list(getattr(context, "selected_objects", [])):
        selected_obj.select_set(False)
    for obj in previous_selected:
        if context.view_layer.objects.get(obj.name) is not None:
            obj.select_set(True)
    if previous_active is not None and context.view_layer.objects.get(previous_active.name) is not None:
        context.view_layer.objects.active = previous_active


def _copy_basic_bone_settings(target_bone, spec):
    target_bone.use_deform = spec.use_deform
    target_bone.roll = spec.roll


def apply_bone_merge(context, source_obj, target_obj, bone_names, use_source_parenting=True):
    """Apply selected source armature rest bones into the target armature."""
    analysis = analyze_bone_merge(source_obj, target_obj, bone_names)
    specs, missing_source_names = _collect_source_specs(source_obj, bone_names)
    if not specs:
        raise ValueError("No source bones are available to apply.")

    previous_active, previous_selected = _save_active_selection(context)
    target_inverse = target_obj.matrix_world.inverted()
    original_target_bone_names = {bone.name for bone in target_obj.data.bones}
    applied_names = [spec.name for spec in specs]
    added_count = 0
    updated_count = 0
    reparented_count = 0
    missing_parent_count = 0

    try:
        _activate_only(context, target_obj)
        bpy.ops.object.mode_set(mode="EDIT")
        target_edit_bones = target_obj.data.edit_bones

        for spec in specs:
            target_bone = target_edit_bones.get(spec.name)
            if target_bone is None:
                target_bone = target_edit_bones.new(spec.name)
                added_count += 1
            else:
                updated_count += 1

            head = target_inverse @ spec.head_world
            tail = target_inverse @ spec.tail_world
            if (tail - head).length <= _MIN_BONE_LENGTH:
                tail.z += _MIN_BONE_LENGTH
            target_bone.head = head
            target_bone.tail = tail
            _copy_basic_bone_settings(target_bone, spec)

        if use_source_parenting:
            available_target_names = {bone.name for bone in target_edit_bones}
            spec_by_name = {spec.name: spec for spec in specs}
            source_spec_names = set(spec_by_name)
            for spec in specs:
                target_bone = target_edit_bones.get(spec.name)
                if target_bone is None:
                    continue

                parent_name = _nearest_available_parent_name(source_obj, spec.name, available_target_names)
                parent_bone = target_edit_bones.get(parent_name) if parent_name else None
                if target_bone.parent != parent_bone:
                    target_bone.parent = parent_bone
                    reparented_count += 1

                can_connect = (
                    parent_name
                    and parent_name == spec.parent_name
                    and parent_name in source_spec_names
                    and spec.use_connect
                )
                target_bone.use_connect = bool(can_connect)
                if parent_name == "" and spec.parent_name:
                    missing_parent_count += 1
    finally:
        if getattr(target_obj, "mode", "OBJECT") != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _restore_active_selection(context, previous_active, previous_selected)

    constraint_counts = _count_constraint_refs_to_bones(
        target_obj,
        [name for name in applied_names if name in original_target_bone_names],
    )
    context.view_layer.update()

    return {
        **analysis,
        "updated_count": updated_count,
        "added_count": added_count,
        "missing_source_count": len(missing_source_names),
        "reparented_count": reparented_count,
        "missing_parent_count": missing_parent_count,
        "constraint_mode": "KEEP_TARGET",
        **constraint_counts,
    }
