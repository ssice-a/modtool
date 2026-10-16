"""Blender background regression tests for applying unlocked shape keys."""

from __future__ import annotations

import sys
from array import array
from pathlib import Path

import bpy


REPOSITORY_PARENT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_PARENT not in sys.path:
    sys.path.insert(0, REPOSITORY_PARENT)

from modtool.core.modifier_apply import apply_unlocked_shape_keys  # noqa: E402


def _reset_scene():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _flat_coordinates(points):
    coordinates = array("f", [0.0]) * (len(points) * 3)
    points.foreach_get("co", coordinates)
    return coordinates


def _evaluated_coordinates(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    try:
        return _flat_coordinates(evaluated_mesh.vertices)
    finally:
        evaluated_obj.to_mesh_clear()


def _offset_shape_key(key_block, offsets):
    coordinates = _flat_coordinates(key_block.data)
    for vertex_index, offset in offsets.items():
        coordinate_index = vertex_index * 3
        coordinates[coordinate_index] += offset[0]
        coordinates[coordinate_index + 1] += offset[1]
        coordinates[coordinate_index + 2] += offset[2]
    key_block.data.foreach_set("co", coordinates)


def _test_apply_unlocked_shape_keys_preserves_locked_mix():
    _reset_scene()
    mesh = bpy.data.meshes.new("ShapeApplyMesh")
    mesh.from_pydata(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (), ((0, 1, 2),))
    obj = bpy.data.objects.new("ShapeApply", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mask = obj.vertex_groups.new(name="Mask")
    mask.add((0,), 1.0, "REPLACE")
    mask.add((2,), 0.5, "REPLACE")

    basis = obj.shape_key_add(name="Basis", from_mix=False)
    unlocked = obj.shape_key_add(name="Unlocked", from_mix=False)
    _offset_shape_key(unlocked, {0: (2.0, 0.0, 0.0), 1: (2.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0)})
    unlocked.vertex_group = mask.name
    unlocked.value = 0.5

    locked = obj.shape_key_add(name="Locked", from_mix=False)
    _offset_shape_key(locked, {0: (0.0, 2.0, 0.0), 1: (0.0, 2.0, 0.0), 2: (0.0, 2.0, 0.0)})
    locked.lock_shape = True
    locked.value = 0.25

    child_locked = obj.shape_key_add(name="ChildLocked", from_mix=False)
    child_locked.relative_key = unlocked
    _offset_shape_key(child_locked, {0: (2.0, 0.0, 4.0), 1: (2.0, 0.0, 4.0), 2: (2.0, 0.0, 4.0)})
    child_locked.lock_shape = True
    child_locked.value = 0.25

    muted = obj.shape_key_add(name="Muted", from_mix=False)
    _offset_shape_key(muted, {0: (0.0, 8.0, 0.0), 1: (0.0, 8.0, 0.0), 2: (0.0, 8.0, 0.0)})
    muted.value = 1.0
    muted.mute = True

    before = _evaluated_coordinates(obj)
    result = apply_unlocked_shape_keys(bpy.context, obj)
    after = _evaluated_coordinates(obj)

    assert result["applied_shape_key_count"] == 2
    assert result["applied_value_count"] == 1
    assert result["locked_shape_key_count"] == 2
    assert list(obj.data.shape_keys.key_blocks.keys()) == ["Basis", "Locked", "ChildLocked"]
    assert obj.data.shape_keys.key_blocks["Locked"].lock_shape
    assert obj.data.shape_keys.key_blocks["ChildLocked"].lock_shape
    assert obj.data.shape_keys.key_blocks["Locked"].relative_key == basis
    assert obj.data.shape_keys.key_blocks["ChildLocked"].relative_key == basis
    assert all(abs(a - b) < 1.0e-6 for a, b in zip(before, after))

    basis_after = _flat_coordinates(obj.data.shape_keys.key_blocks["Basis"].data)
    assert abs(basis_after[0] - 1.0) < 1.0e-6
    assert abs(basis_after[3] - 0.0) < 1.0e-6
    assert abs(basis_after[6] - 0.5) < 1.0e-6
    print("APPLY_UNLOCKED_SHAPE_KEYS_OK")


if __name__ == "__main__":
    _test_apply_unlocked_shape_keys_preserves_locked_mix()
