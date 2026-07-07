"""Modifier-apply helpers that preserve shape keys."""

from __future__ import annotations

from array import array

import bpy

from .common import get_flat_point_coordinates, is_mesh_object, sync_edit_mesh_data


try:
    import numpy as _np
except Exception:
    _np = None


_SHAPE_KEY_VALUE_EPSILON = 1.0e-8


def has_shape_keys(obj):
    """Return True when the mesh owns any shape keys, including Basis."""
    if not is_mesh_object(obj):
        return False

    shape_keys = getattr(obj.data, "shape_keys", None)
    return shape_keys is not None and len(shape_keys.key_blocks) > 0


def get_enabled_modifier_names(obj):
    """Return the viewport-enabled modifiers on the object in stack order."""
    if not is_mesh_object(obj):
        return []
    return [modifier.name for modifier in obj.modifiers if modifier.show_viewport]


def _shape_key_blocks(obj):
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None or len(shape_keys.key_blocks) == 0:
        return None, []
    return shape_keys, list(shape_keys.key_blocks)


def _non_basis_value_count(shape_key_blocks):
    return sum(1 for key_block in shape_key_blocks[1:] if abs(float(key_block.value)) > _SHAPE_KEY_VALUE_EPSILON)


def _collect_shape_key_metadata(shape_keys):
    metadata = []
    for index, key_block in enumerate(shape_keys.key_blocks):
        metadata.append(
            {
                "index": index,
                "name": key_block.name,
                "value": key_block.value,
                "mute": key_block.mute,
                "lock_shape": bool(getattr(key_block, "lock_shape", False)),
                "slider_min": key_block.slider_min,
                "slider_max": key_block.slider_max,
                "vertex_group": key_block.vertex_group,
                "interpolation": key_block.interpolation,
                "relative_name": key_block.relative_key.name if key_block.relative_key is not None else "",
                "frame": getattr(key_block, "frame", 0.0),
            }
        )

    return {
        "name": shape_keys.name,
        "use_relative": bool(shape_keys.use_relative),
        "eval_time": getattr(shape_keys, "eval_time", 0.0),
        "key_blocks": metadata,
    }


def _capture_runtime_state(obj):
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None:
        return None

    return {
        "active_shape_key_index": getattr(obj, "active_shape_key_index", 0),
        "show_only_shape_key": bool(getattr(obj, "show_only_shape_key", False)),
        "eval_time": getattr(shape_keys, "eval_time", 0.0),
        "key_values": {
            key_block.name: {"value": key_block.value, "mute": key_block.mute}
            for key_block in shape_keys.key_blocks
        },
    }


def _restore_runtime_state(obj, runtime_state):
    if runtime_state is None:
        return

    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None:
        return

    for key_block in shape_keys.key_blocks:
        key_state = runtime_state["key_values"].get(key_block.name)
        if key_state is None:
            continue
        key_block.value = key_state["value"]
        key_block.mute = key_state["mute"]

    if hasattr(shape_keys, "eval_time"):
        shape_keys.eval_time = runtime_state["eval_time"]
    obj.show_only_shape_key = runtime_state["show_only_shape_key"]
    obj.active_shape_key_index = min(
        max(0, runtime_state["active_shape_key_index"]),
        max(0, len(shape_keys.key_blocks) - 1),
    )


def _capture_modifier_states(obj):
    return {
        modifier.name: modifier.show_viewport
        for modifier in obj.modifiers
    }


def _restore_modifier_states(obj, modifier_states, skip_names=None):
    skip_names = set() if skip_names is None else set(skip_names)
    for modifier in obj.modifiers:
        if modifier.name in skip_names:
            continue
        if modifier.name not in modifier_states:
            continue
        modifier.show_viewport = modifier_states[modifier.name]


def _resolve_modifier_prefix(obj, modifier_name):
    if modifier_name:
        target_index = obj.modifiers.find(modifier_name)
        if target_index < 0:
            raise ValueError("Choose a valid modifier to apply.")
    else:
        target_index = -1
        for modifier_index, modifier in enumerate(obj.modifiers):
            if modifier.show_viewport:
                target_index = modifier_index

        if target_index < 0:
            raise ValueError("The object has no viewport-enabled modifiers.")

    apply_modifier_names = []
    for modifier_index, modifier in enumerate(obj.modifiers):
        if modifier_index > target_index:
            break
        if modifier.show_viewport:
            apply_modifier_names.append(modifier.name)

    if modifier_name and modifier_name not in apply_modifier_names:
        raise ValueError("Choose a viewport-enabled modifier. Disabled modifiers are not applied.")
    if not apply_modifier_names:
        raise ValueError("The object has no viewport-enabled modifiers.")

    return target_index, apply_modifier_names


def _set_capture_modifier_prefix(obj, target_index):
    for modifier_index, modifier in enumerate(obj.modifiers):
        if modifier_index > target_index:
            modifier.show_viewport = False


def _prepare_shape_key_preview_capture(obj):
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None or len(shape_keys.key_blocks) == 0:
        return None

    for key_block in shape_keys.key_blocks:
        key_block.value = 0.0
        key_block.mute = False

    obj.show_only_shape_key = True
    return shape_keys


def _set_preview_shape(obj, shape_keys, key_index):
    if shape_keys is None or len(shape_keys.key_blocks) == 0:
        return

    obj.active_shape_key_index = min(max(0, key_index), len(shape_keys.key_blocks) - 1)
    if not shape_keys.use_relative and hasattr(shape_keys, "eval_time"):
        key_block = shape_keys.key_blocks[obj.active_shape_key_index]
        if hasattr(key_block, "frame"):
            shape_keys.eval_time = key_block.frame


def _capture_flat_vertex_coordinates(obj, depsgraph, expected_vertex_count=None):
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    try:
        vertex_count = len(evaluated_mesh.vertices)
        if expected_vertex_count is not None and vertex_count != expected_vertex_count:
            raise ValueError(
                "The selected modifier stack changes topology between shape keys. "
                f"Expected {expected_vertex_count} vertices, got {vertex_count}."
            )

        flat_coordinates = get_flat_point_coordinates(evaluated_mesh.vertices)
        return flat_coordinates, vertex_count
    finally:
        evaluated_obj.to_mesh_clear()


def _create_evaluated_mesh_snapshot(obj, depsgraph):
    evaluated_obj = obj.evaluated_get(depsgraph)
    applied_mesh = bpy.data.meshes.new_from_object(
        evaluated_obj,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    if applied_mesh is None:
        raise ValueError("Could not build the evaluated modifier result.")
    applied_mesh.name = obj.data.name
    return applied_mesh


def _apply_flat_coordinates_to_mesh(mesh, flat_coordinates):
    mesh.vertices.foreach_set("co", flat_coordinates)
    mesh.update()


def _copy_action_settings(source_anim, target_anim):
    target_anim.action = source_anim.action
    target_anim.action_blend_type = source_anim.action_blend_type
    target_anim.action_extrapolation = source_anim.action_extrapolation
    target_anim.action_influence = source_anim.action_influence
    target_anim.use_nla = source_anim.use_nla


def _copy_driver_variable(source_variable, target_variable):
    target_variable.name = source_variable.name
    target_variable.type = source_variable.type
    for source_target, target_target in zip(source_variable.targets, target_variable.targets):
        for attribute_name in (
            "id",
            "id_type",
            "bone_target",
            "data_path",
            "transform_type",
            "transform_space",
            "rotation_mode",
            "context_property",
            "use_fallback_value",
            "fallback_value",
        ):
            if hasattr(source_target, attribute_name) and hasattr(target_target, attribute_name):
                setattr(target_target, attribute_name, getattr(source_target, attribute_name))


def _copy_shape_key_drivers(source_shape_keys, target_shape_keys):
    source_anim = getattr(source_shape_keys, "animation_data", None)
    if source_anim is None or len(source_anim.drivers) == 0:
        return 0

    target_anim = target_shape_keys.animation_data_create()
    copied_count = 0
    for source_fcurve in source_anim.drivers:
        try:
            target_fcurve = target_shape_keys.driver_add(source_fcurve.data_path, source_fcurve.array_index)
        except TypeError:
            continue
        except ValueError:
            continue

        target_fcurve.mute = source_fcurve.mute
        target_fcurve.lock = source_fcurve.lock
        target_fcurve.extrapolation = source_fcurve.extrapolation

        source_driver = source_fcurve.driver
        target_driver = target_fcurve.driver
        target_driver.type = source_driver.type
        target_driver.expression = source_driver.expression
        target_driver.use_self = source_driver.use_self

        while len(target_driver.variables) > 0:
            target_driver.variables.remove(target_driver.variables[0])

        for source_variable in source_driver.variables:
            target_variable = target_driver.variables.new()
            _copy_driver_variable(source_variable, target_variable)

        copied_count += 1

    return copied_count


def _copy_shape_key_animation(source_shape_keys, target_shape_keys):
    source_anim = getattr(source_shape_keys, "animation_data", None)
    if source_anim is None:
        return {"action_copied": False, "driver_count": 0}

    target_anim = target_shape_keys.animation_data_create()
    _copy_action_settings(source_anim, target_anim)
    return {
        "action_copied": source_anim.action is not None,
        "driver_count": _copy_shape_key_drivers(source_shape_keys, target_shape_keys),
    }


def _apply_shape_key_metadata(key_block, key_info):
    key_block.slider_min = key_info["slider_min"]
    key_block.slider_max = key_info["slider_max"]
    key_block.value = key_info["value"]
    key_block.mute = key_info["mute"]
    if hasattr(key_block, "lock_shape"):
        key_block.lock_shape = key_info["lock_shape"]
    key_block.vertex_group = key_info["vertex_group"]
    key_block.interpolation = key_info["interpolation"]
    if hasattr(key_block, "frame"):
        try:
            key_block.frame = key_info["frame"]
        except (AttributeError, TypeError):
            # Blender 5.0 exposes ShapeKey.frame as read-only.
            pass


def _rebuild_shape_keys(obj, shape_key_info, flat_key_coordinates):
    basis_info = shape_key_info["key_blocks"][0]
    basis_key = obj.shape_key_add(name=basis_info["name"], from_mix=False)
    rebuilt_keys = {basis_key.name: basis_key}

    for key_info, flat_coordinates in zip(shape_key_info["key_blocks"][1:], flat_key_coordinates):
        key_block = obj.shape_key_add(name=key_info["name"], from_mix=False)
        target_points = getattr(key_block, "points", key_block.data)
        target_points.foreach_set("co", flat_coordinates)
        _apply_shape_key_metadata(key_block, key_info)
        rebuilt_keys[key_block.name] = key_block

    shape_keys = obj.data.shape_keys
    shape_keys.name = shape_key_info["name"]
    shape_keys.use_relative = shape_key_info["use_relative"]
    if hasattr(shape_keys, "eval_time"):
        shape_keys.eval_time = shape_key_info["eval_time"]

    for key_info in shape_key_info["key_blocks"]:
        key_block = rebuilt_keys.get(key_info["name"])
        if key_block is None:
            continue
        relative_name = key_info["relative_name"]
        if not relative_name or relative_name == basis_key.name:
            key_block.relative_key = basis_key
            continue
        key_block.relative_key = rebuilt_keys.get(relative_name, basis_key)

    obj.data.update()
    return rebuilt_keys


def _restore_new_runtime_state(obj, runtime_state):
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None or runtime_state is None:
        return

    for key_block in shape_keys.key_blocks:
        key_state = runtime_state["key_values"].get(key_block.name)
        if key_state is None:
            continue
        key_block.value = key_state["value"]
        key_block.mute = key_state["mute"]

    if hasattr(shape_keys, "eval_time"):
        shape_keys.eval_time = runtime_state["eval_time"]
    obj.show_only_shape_key = runtime_state["show_only_shape_key"]
    obj.active_shape_key_index = min(
        max(0, runtime_state["active_shape_key_index"]),
        len(shape_keys.key_blocks) - 1,
    )


def _flat_delta_coordinates(key_coordinates, basis_coordinates):
    if _np is not None:
        return _np.asarray(key_coordinates, dtype=_np.float32) - _np.asarray(basis_coordinates, dtype=_np.float32)

    return array(
        "f",
        (
            float(key_coordinates[index]) - float(basis_coordinates[index])
            for index in range(len(basis_coordinates))
        ),
    )


def _flat_add_coordinates(coordinates_a, coordinates_b):
    if _np is not None:
        return _np.asarray(coordinates_a, dtype=_np.float32) + _np.asarray(coordinates_b, dtype=_np.float32)

    return array(
        "f",
        (
            float(coordinates_a[index]) + float(coordinates_b[index])
            for index in range(len(coordinates_a))
        ),
    )


def _flat_accumulate_scaled(base_coordinates, delta_coordinates, scale):
    if abs(scale) <= _SHAPE_KEY_VALUE_EPSILON:
        return base_coordinates
    if _np is not None:
        return _np.asarray(base_coordinates, dtype=_np.float32) + (
            _np.asarray(delta_coordinates, dtype=_np.float32) * float(scale)
        )

    return array(
        "f",
        (
            float(base_coordinates[index]) + (float(delta_coordinates[index]) * float(scale))
            for index in range(len(base_coordinates))
        ),
    )


def _flat_scale_coordinates(coordinates, scale):
    if abs(scale - 1.0) <= _SHAPE_KEY_VALUE_EPSILON:
        return coordinates
    if _np is not None:
        return _np.asarray(coordinates, dtype=_np.float32) * float(scale)

    return array("f", (float(coordinate) * float(scale) for coordinate in coordinates))


def _flat_masked_coordinates(coordinates, weights):
    if weights is None:
        return coordinates
    if _np is not None:
        coordinate_array = _np.asarray(coordinates, dtype=_np.float32).reshape((-1, 3))
        weight_array = _np.asarray(weights, dtype=_np.float32).reshape((-1, 1))
        return (coordinate_array * weight_array).reshape(-1).astype(_np.float32)

    masked_coordinates = array("f", [0.0]) * len(coordinates)
    for vertex_index, weight in enumerate(weights):
        coordinate_index = vertex_index * 3
        masked_coordinates[coordinate_index] = float(coordinates[coordinate_index]) * float(weight)
        masked_coordinates[coordinate_index + 1] = float(coordinates[coordinate_index + 1]) * float(weight)
        masked_coordinates[coordinate_index + 2] = float(coordinates[coordinate_index + 2]) * float(weight)
    return masked_coordinates


def _flat_vertex_group_weights(obj, group_name, vertex_count, weight_cache):
    if not group_name:
        return None
    if group_name in weight_cache:
        return weight_cache[group_name]

    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        raise ValueError(f"Shape key vertex group does not exist: {group_name}.")

    if _np is not None:
        weights = _np.zeros(vertex_count, dtype=_np.float32)
    else:
        weights = array("f", [0.0]) * vertex_count

    group_index = vertex_group.index
    for vertex in obj.data.vertices:
        for membership in vertex.groups:
            if membership.group != group_index:
                continue
            weights[vertex.index] = membership.weight
            break

    weight_cache[group_name] = weights
    return weights


def _shape_key_basis_delta(obj, key_block, delta_coordinates, vertex_count, weight_cache):
    weights = _flat_vertex_group_weights(obj, key_block.vertex_group, vertex_count, weight_cache)
    return _flat_masked_coordinates(delta_coordinates, weights)


def _ensure_shape_key_range_remap_ready(obj, shape_keys, key_blocks):
    if shape_keys is None or len(key_blocks) <= 1:
        raise ValueError("The object needs at least one non-Basis shape key.")
    if not getattr(shape_keys, "use_relative", True):
        raise ValueError("Shape-key range remapping only supports relative shape keys.")
    if getattr(obj, "show_only_shape_key", False):
        raise ValueError("Disable solo shape-key view before remapping shape-key ranges.")

    muted_key_names = [key_block.name for key_block in key_blocks[1:] if key_block.mute]
    if muted_key_names:
        preview_names = ", ".join(muted_key_names[:3])
        if len(muted_key_names) > 3:
            preview_names += f" (+{len(muted_key_names) - 3})"
        raise ValueError(f"Unmute shape keys before remapping ranges: {preview_names}.")


def _collect_relative_shape_key_coordinate_data(key_blocks):
    key_coordinates_by_name = {
        key_block.name: get_flat_point_coordinates(key_block.data)
        for key_block in key_blocks
    }
    basis_name = key_blocks[0].name
    basis_coordinates = key_coordinates_by_name[basis_name]
    delta_items = []

    for key_block in key_blocks[1:]:
        relative_key = key_block.relative_key if key_block.relative_key is not None else key_blocks[0]
        relative_name = relative_key.name
        relative_coordinates = key_coordinates_by_name.get(relative_name, basis_coordinates)
        delta_items.append(
            {
                "name": key_block.name,
                "relative_name": relative_name,
                "delta_coordinates": _flat_delta_coordinates(
                    key_coordinates_by_name[key_block.name],
                    relative_coordinates,
                ),
            }
        )

    return {
        "basis_name": basis_name,
        "basis_coordinates": basis_coordinates,
        "delta_items": delta_items,
    }


def _rebuild_relative_shape_key_coordinates(basis_coordinates, basis_name, delta_items):
    rebuilt_coordinates_by_name = {basis_name: basis_coordinates}
    pending_items = list(delta_items)

    while pending_items:
        next_pending_items = []
        progressed = False
        for item in pending_items:
            relative_coordinates = rebuilt_coordinates_by_name.get(item["relative_name"])
            if relative_coordinates is None:
                next_pending_items.append(item)
                continue

            rebuilt_coordinates_by_name[item["name"]] = _flat_add_coordinates(
                relative_coordinates,
                item["delta_coordinates"],
            )
            progressed = True

        if not progressed:
            for item in next_pending_items:
                rebuilt_coordinates_by_name[item["name"]] = _flat_add_coordinates(
                    basis_coordinates,
                    item["delta_coordinates"],
                )
            break
        pending_items = next_pending_items

    return rebuilt_coordinates_by_name


def _write_shape_key_coordinates(key_blocks, coordinates_by_name):
    for key_block in key_blocks:
        coordinates = coordinates_by_name.get(key_block.name)
        if coordinates is None:
            continue
        target_points = getattr(key_block, "points", key_block.data)
        target_points.foreach_set("co", coordinates)


def _shape_key_range_width(key_block):
    return float(key_block.slider_max) - float(key_block.slider_min)


def set_current_shape_key_values_as_zero(context, obj):
    """Make the current relative shape-key mix the new zero point without changing the visible mesh."""
    if not is_mesh_object(obj):
        raise ValueError("Choose a mesh object.")

    sync_edit_mesh_data(obj)
    shape_keys, key_blocks = _shape_key_blocks(obj)
    _ensure_shape_key_range_remap_ready(obj, shape_keys, key_blocks)

    coordinate_data = _collect_relative_shape_key_coordinate_data(key_blocks)
    basis_coordinates = coordinate_data["basis_coordinates"]
    zero_basis_coordinates = basis_coordinates
    nonzero_value_count = _non_basis_value_count(key_blocks)
    value_by_name = {}
    weight_cache = {}
    vertex_count = len(key_blocks[0].data)

    for key_block, item in zip(key_blocks[1:], coordinate_data["delta_items"]):
        key_value = float(key_block.value)
        value_by_name[key_block.name] = key_value
        basis_delta_coordinates = _shape_key_basis_delta(
            obj,
            key_block,
            item["delta_coordinates"],
            vertex_count,
            weight_cache,
        )
        zero_basis_coordinates = _flat_accumulate_scaled(
            zero_basis_coordinates,
            basis_delta_coordinates,
            key_value,
        )

    rebuilt_coordinates = _rebuild_relative_shape_key_coordinates(
        zero_basis_coordinates,
        coordinate_data["basis_name"],
        coordinate_data["delta_items"],
    )
    _write_shape_key_coordinates(key_blocks, rebuilt_coordinates)

    for key_block in key_blocks[1:]:
        key_value = value_by_name[key_block.name]
        key_block.slider_min = float(key_block.slider_min) - key_value
        key_block.slider_max = float(key_block.slider_max) - key_value
        key_block.value = 0.0

    obj.data.update()
    context.view_layer.update()

    return {
        "object_name": obj.name,
        "shape_key_count": max(0, len(key_blocks) - 1),
        "nonzero_value_count": nonzero_value_count,
        "uses_numpy": _np is not None,
    }


def normalize_shape_key_ranges(context, obj):
    """Map every relative shape-key slider range to 0..1 without changing the visible mesh."""
    if not is_mesh_object(obj):
        raise ValueError("Choose a mesh object.")

    sync_edit_mesh_data(obj)
    shape_keys, key_blocks = _shape_key_blocks(obj)
    _ensure_shape_key_range_remap_ready(obj, shape_keys, key_blocks)

    zero_width_keys = [
        key_block.name
        for key_block in key_blocks[1:]
        if _shape_key_range_width(key_block) <= _SHAPE_KEY_VALUE_EPSILON
    ]
    if zero_width_keys:
        preview_names = ", ".join(zero_width_keys[:3])
        if len(zero_width_keys) > 3:
            preview_names += f" (+{len(zero_width_keys) - 3})"
        raise ValueError(f"Cannot normalize zero-width shape-key ranges: {preview_names}.")

    coordinate_data = _collect_relative_shape_key_coordinate_data(key_blocks)
    normalized_basis_coordinates = coordinate_data["basis_coordinates"]
    normalized_delta_items = []
    metadata_by_name = {}
    weight_cache = {}
    vertex_count = len(key_blocks[0].data)

    for key_block, item in zip(key_blocks[1:], coordinate_data["delta_items"]):
        slider_min = float(key_block.slider_min)
        slider_max = float(key_block.slider_max)
        range_width = slider_max - slider_min
        metadata_by_name[key_block.name] = {
            "value": float(key_block.value),
            "slider_min": slider_min,
            "range_width": range_width,
        }
        basis_delta_coordinates = _shape_key_basis_delta(
            obj,
            key_block,
            item["delta_coordinates"],
            vertex_count,
            weight_cache,
        )
        normalized_basis_coordinates = _flat_accumulate_scaled(
            normalized_basis_coordinates,
            basis_delta_coordinates,
            slider_min,
        )
        normalized_delta_items.append(
            {
                "name": item["name"],
                "relative_name": item["relative_name"],
                "delta_coordinates": _flat_scale_coordinates(item["delta_coordinates"], range_width),
            }
        )

    rebuilt_coordinates = _rebuild_relative_shape_key_coordinates(
        normalized_basis_coordinates,
        coordinate_data["basis_name"],
        normalized_delta_items,
    )
    _write_shape_key_coordinates(key_blocks, rebuilt_coordinates)

    changed_range_count = 0
    for key_block in key_blocks[1:]:
        metadata = metadata_by_name[key_block.name]
        normalized_value = (metadata["value"] - metadata["slider_min"]) / metadata["range_width"]
        if (
            abs(float(key_block.slider_min)) > _SHAPE_KEY_VALUE_EPSILON
            or abs(float(key_block.slider_max) - 1.0) > _SHAPE_KEY_VALUE_EPSILON
        ):
            changed_range_count += 1
        key_block.slider_min = 0.0
        key_block.slider_max = 1.0
        key_block.value = normalized_value

    obj.data.update()
    context.view_layer.update()

    return {
        "object_name": obj.name,
        "shape_key_count": max(0, len(key_blocks) - 1),
        "changed_range_count": changed_range_count,
        "uses_numpy": _np is not None,
    }


def apply_modifier_with_shape_keys(context, obj, modifier_name=""):
    """Apply enabled modifiers through the chosen modifier while preserving shape keys."""
    if not is_mesh_object(obj):
        raise ValueError("Choose a mesh object.")

    sync_edit_mesh_data(obj)
    if not has_shape_keys(obj):
        raise ValueError("The object has no shape keys to preserve.")
    if len(obj.modifiers) == 0:
        raise ValueError("The object has no modifiers.")

    target_index, apply_modifier_names = _resolve_modifier_prefix(obj, modifier_name)
    original_modifier_states = _capture_modifier_states(obj)
    runtime_state = _capture_runtime_state(obj)
    old_mesh = obj.data
    old_shape_keys = getattr(old_mesh, "shape_keys", None)
    shape_key_info = _collect_shape_key_metadata(old_shape_keys)
    depsgraph = context.evaluated_depsgraph_get()
    flat_key_coordinates = []
    applied_mesh = None
    action_restore_info = {"action_copied": False, "driver_count": 0}

    try:
        _set_capture_modifier_prefix(obj, target_index)
        context.view_layer.update()

        applied_mesh = _create_evaluated_mesh_snapshot(obj, depsgraph)
        expected_vertex_count = len(applied_mesh.vertices)

        preview_shape_keys = _prepare_shape_key_preview_capture(obj)
        _set_preview_shape(obj, preview_shape_keys, 0)
        context.view_layer.update()
        basis_coordinates, _basis_vertex_count = _capture_flat_vertex_coordinates(
            obj,
            depsgraph,
            expected_vertex_count=expected_vertex_count,
        )
        _apply_flat_coordinates_to_mesh(applied_mesh, basis_coordinates)

        for key_info in shape_key_info["key_blocks"][1:]:
            _set_preview_shape(obj, preview_shape_keys, key_info["index"])
            context.view_layer.update()
            flat_coordinates, _vertex_count = _capture_flat_vertex_coordinates(
                obj,
                depsgraph,
                expected_vertex_count=expected_vertex_count,
            )
            flat_key_coordinates.append(flat_coordinates)
    finally:
        _restore_runtime_state(obj, runtime_state)
        _restore_modifier_states(obj, original_modifier_states)
        context.view_layer.update()

    if applied_mesh is None:
        raise ValueError("Could not build the evaluated modifier result.")

    success = False
    try:
        for modifier in obj.modifiers:
            if modifier.name in apply_modifier_names:
                modifier.show_viewport = False
        context.view_layer.update()

        obj.data = applied_mesh
        _rebuild_shape_keys(obj, shape_key_info, flat_key_coordinates)
        _restore_new_runtime_state(obj, runtime_state)

        if old_shape_keys is not None and obj.data.shape_keys is not None:
            action_restore_info = _copy_shape_key_animation(old_shape_keys, obj.data.shape_keys)

        for modifier_name_to_remove in reversed(apply_modifier_names):
            modifier = obj.modifiers.get(modifier_name_to_remove)
            if modifier is not None:
                obj.modifiers.remove(modifier)

        _restore_modifier_states(obj, original_modifier_states)
        context.view_layer.update()
        success = True
    finally:
        if not success:
            if obj.data == applied_mesh:
                obj.data = old_mesh
            _restore_runtime_state(obj, runtime_state)
            _restore_modifier_states(obj, original_modifier_states)
            context.view_layer.update()
            if applied_mesh is not None and applied_mesh.users == 0:
                bpy.data.meshes.remove(applied_mesh)

    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    return {
        "applied_modifier_names": apply_modifier_names,
        "shape_key_count": max(0, len(shape_key_info["key_blocks"]) - 1),
        "vertex_count": len(obj.data.vertices),
        "vertex_group_count": len(obj.vertex_groups),
        "action_copied": action_restore_info["action_copied"],
        "driver_count": action_restore_info["driver_count"],
    }
