"""Shared Blender helpers used across Mod Tool features."""

from __future__ import annotations

from array import array

import bpy
from mathutils import Vector


_MODE_RESTORE_MAP = {
    "EDIT_ARMATURE": "EDIT",
    "EDIT_MESH": "EDIT",
    "OBJECT": "OBJECT",
    "POSE": "POSE",
    "SCULPT": "SCULPT",
    "VERTEX_PAINT": "VERTEX_PAINT",
    "WEIGHT_PAINT": "WEIGHT_PAINT",
    "TEXTURE_PAINT": "TEXTURE_PAINT",
}


def is_mesh_object(obj):
    """Return True when the object is a mesh."""
    return obj is not None and getattr(obj, "type", None) == "MESH"


def is_armature_object(obj):
    """Return True when the object is an armature."""
    return obj is not None and getattr(obj, "type", None) == "ARMATURE"


def get_selected_mesh_objects(context):
    """Return every selected mesh object in the current view layer."""
    return [obj for obj in getattr(context, "selected_objects", []) if is_mesh_object(obj)]


def sync_edit_mesh_data(obj):
    """Push edit-mode changes back to the mesh before reading geometry data."""
    if obj is not None and getattr(obj, "mode", "OBJECT") == "EDIT":
        obj.update_from_editmode()


def get_flat_point_coordinates(points):
    """Read local point coordinates with Blender's bulk RNA API."""
    flat_coordinates = array("f", [0.0]) * (len(points) * 3)
    points.foreach_get("co", flat_coordinates)
    return flat_coordinates


def get_point_coordinates(points):
    """Read local point coordinates as mathutils vectors."""
    flat_coordinates = get_flat_point_coordinates(points)
    return [
        Vector(
            (
                flat_coordinates[coordinate_index],
                flat_coordinates[coordinate_index + 1],
                flat_coordinates[coordinate_index + 2],
            )
        )
        for coordinate_index in range(0, len(flat_coordinates), 3)
    ]


def get_world_point_coordinates(points, matrix_world):
    """Read point coordinates and transform them into world space."""
    flat_coordinates = get_flat_point_coordinates(points)
    return [
        matrix_world
        @ Vector(
            (
                flat_coordinates[coordinate_index],
                flat_coordinates[coordinate_index + 1],
                flat_coordinates[coordinate_index + 2],
            )
        )
        for coordinate_index in range(0, len(flat_coordinates), 3)
    ]


def ensure_object_mode(context):
    """Switch to Object Mode and return the previous mode."""
    previous_mode = context.mode
    if previous_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    return previous_mode


def restore_context_mode(previous_mode):
    """Restore the previous interaction mode when possible."""
    restore_mode = _MODE_RESTORE_MAP.get(previous_mode)
    if restore_mode is None or restore_mode == "OBJECT":
        return

    try:
        bpy.ops.object.mode_set(mode=restore_mode)
    except RuntimeError:
        pass


def run_object_operator(context, active_obj, operator_name, **kwargs):
    """Run an object operator with a temporary active selection and restore the user's selection."""
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(getattr(context, "selected_objects", []))

    try:
        for obj in previous_selected:
            obj.select_set(False)
        active_obj.select_set(True)
        view_layer.objects.active = active_obj
        return getattr(bpy.ops.object, operator_name)(**kwargs)
    finally:
        for obj in list(getattr(context, "selected_objects", [])):
            obj.select_set(False)
        for obj in previous_selected:
            if view_layer.objects.get(obj.name) is not None:
                obj.select_set(True)
        if previous_active is not None and view_layer.objects.get(previous_active.name) is not None:
            view_layer.objects.active = previous_active
