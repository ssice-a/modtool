"""Shared Blender helpers used across Mod Tool features."""

from __future__ import annotations

from array import array
import hashlib

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
    "PAINT_WEIGHT": "WEIGHT_PAINT",
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


def mesh_topology_signature(mesh, include_edges=False):
    """Hash vertex identities and polygon connectivity with bulk RNA reads."""
    digest = hashlib.sha256()
    digest.update(len(mesh.vertices).to_bytes(8, "little"))
    digest.update(len(mesh.edges).to_bytes(8, "little"))
    for elements, field, size in (
        (mesh.loops, "vertex_index", len(mesh.loops)),
        (mesh.polygons, "loop_start", len(mesh.polygons)),
        (mesh.polygons, "loop_total", len(mesh.polygons)),
    ):
        values = array("i", [0]) * size
        if size:
            elements.foreach_get(field, values)
        digest.update(values.tobytes())
    if include_edges:
        values = array("i", [0]) * (len(mesh.edges) * 2)
        if values:
            mesh.edges.foreach_get("vertices", values)
        digest.update(values.tobytes())
    return digest.hexdigest()


def get_flat_point_coordinates(points):
    """Read local point coordinates with Blender's bulk RNA API."""
    flat_coordinates = array("f", [0.0]) * (len(points) * 3)
    points.foreach_get("co", flat_coordinates)
    return flat_coordinates


def mesh_coordinate_offset_plan(mesh, delta, *, coordinates=None, keys=None):
    """Prepare rollback-capable bulk writes preserving every shape-key delta."""
    from .materials import _require_numpy, np

    _require_numpy("Mesh coordinate offsets")
    if coordinates is None:
        coordinates = np.frombuffer(get_flat_point_coordinates(mesh.vertices), dtype=np.float32).reshape(-1, 3)
    if keys is None:
        keys = [(key, np.frombuffer(get_flat_point_coordinates(key.data), dtype=np.float32).reshape(-1, 3))
                for key in mesh.shape_keys.key_blocks] if mesh.shape_keys else []
    targets = [(mesh.vertices, coordinates, coordinates + delta)]
    targets.extend((key.data, values, values + delta) for key, values in keys)
    return targets


def triangle_barycentric_array(points, triangles):
    """Batch barycentric coordinates for surface hits; return usable rows too."""
    from .materials import np

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, offset = b - a, c - a, points - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    denominator = d00 * d11 - d01 * d01
    usable = np.abs(denominator) > 1e-20
    wb = np.zeros(len(points)); wc = np.zeros(len(points))
    d20, d21 = dot(offset, ab), dot(offset, ac)
    wb[usable] = (d11[usable] * d20[usable] - d01[usable] * d21[usable]) / denominator[usable]
    wc[usable] = (d00[usable] * d21[usable] - d01[usable] * d20[usable]) / denominator[usable]
    values = np.column_stack((1.0 - wb - wc, wb, wc))
    np.clip(values, 0.0, 1.0, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values, usable


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
