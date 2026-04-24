"""Shared Blender helpers used across Mod Tool features."""

from __future__ import annotations

import bpy


_MODE_RESTORE_MAP = {
    "EDIT_MESH": "EDIT",
    "OBJECT": "OBJECT",
    "SCULPT": "SCULPT",
    "VERTEX_PAINT": "VERTEX_PAINT",
    "WEIGHT_PAINT": "WEIGHT_PAINT",
    "TEXTURE_PAINT": "TEXTURE_PAINT",
}


def is_mesh_object(obj):
    """Return True when the object is a mesh."""
    return obj is not None and getattr(obj, "type", None) == "MESH"


def get_selected_mesh_objects(context):
    """Return every selected mesh object in the current view layer."""
    return [obj for obj in getattr(context, "selected_objects", []) if is_mesh_object(obj)]


def sync_edit_mesh_data(obj):
    """Push edit-mode changes back to the mesh before reading geometry data."""
    if obj is not None and getattr(obj, "mode", "OBJECT") == "EDIT":
        obj.update_from_editmode()


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
