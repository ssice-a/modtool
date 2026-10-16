"""Mixed-selection modifier bake: visible geometry, keys and context preservation."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np

from modtool.ops import modifier_apply as operators


def coordinates(points):
    values = np.empty((len(points), 3), dtype=np.float32)
    points.foreach_get("co", values.ravel())
    return values


def visible(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return coordinates(mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def fixture():
    bpy.ops.mesh.primitive_plane_add()
    keyed = bpy.context.object
    keyed.name = "BatchKeyed"
    keyed.shape_key_add(name="Basis")
    key = keyed.shape_key_add(name="Raised")
    co = coordinates(key.data)
    co[0, 2] = .4
    key.data.foreach_set("co", co.ravel())
    key.value = .35
    key.lock_shape = True
    keyed.modifiers.new("Subdivide", "SUBSURF").levels = 1

    bpy.ops.mesh.primitive_plane_add()
    plain = bpy.context.object
    plain.name = "BatchPlain"
    plain.modifiers.new("Thickness", "SOLIDIFY").thickness = .2
    plain.modifiers.new("Disabled", "SUBSURF").show_viewport = False
    shared = plain.copy()
    bpy.context.collection.objects.link(shared)
    shared.name = "UnselectedSharedMesh"

    bpy.ops.mesh.primitive_plane_add()
    skipped = bpy.context.object
    skipped.name = "NoModifiers"
    empty = bpy.data.objects.new("SelectedEmpty", None)
    bpy.context.collection.objects.link(empty)
    bpy.context.view_layer.update()
    return keyed, plain, shared, skipped, empty


def select(objects, active):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def run():
    bpy.utils.register_class(operators.MDT_OT_apply_modifier_keep_shape_keys)
    keyed, plain, shared, skipped, empty = fixture()
    before = {obj.name: visible(obj) for obj in (keyed, plain)}
    original_shared_data = shared.data
    shared_coordinates = coordinates(shared.data.vertices)
    chosen = [keyed, plain, skipped, empty]
    select(chosen, empty)
    assert bpy.ops.mdt.apply_modifier_keep_shape_keys() == {"FINISHED"}
    for obj in (keyed, plain):
        np.testing.assert_allclose(visible(obj), before[obj.name], atol=2e-6)
    assert not keyed.modifiers
    assert [m.name for m in plain.modifiers] == ["Disabled"]
    assert not plain.modifiers[0].show_viewport
    assert plain.data.shape_keys is None
    assert list(keyed.data.shape_keys.key_blocks.keys()) == ["Basis", "Raised"]
    key = keyed.data.shape_keys.key_blocks["Raised"]
    assert abs(key.value - .35) < 1e-6 and key.lock_shape
    assert shared.data == original_shared_data and plain.data != shared.data
    np.testing.assert_array_equal(coordinates(shared.data.vertices), shared_coordinates)
    assert len(shared.modifiers) == 2
    assert set(bpy.context.selected_objects) == set(chosen)
    assert bpy.context.object == empty

    # The active object must stay active and multi-object Edit Mode must return.
    for obj in (keyed, plain):
        obj.modifiers.new("NextSubdivision", "SUBSURF").levels = 1
    select([keyed, plain], plain)
    bpy.ops.object.mode_set(mode="EDIT")
    assert bpy.ops.mdt.apply_modifier_keep_shape_keys() == {"FINISHED"}
    assert bpy.context.mode == "EDIT_MESH" and bpy.context.object == plain
    assert set(bpy.context.objects_in_mode) == {keyed, plain}
    bpy.ops.object.mode_set(mode="OBJECT")

    # A failing object must not stop the rest of the selection from being baked.
    for obj in (keyed, plain):
        obj.modifiers.new("FinalSubdivision", "SUBSURF").levels = 1
    original_apply = operators.apply_enabled_modifiers
    def fail_one(context, obj):
        if obj == keyed:
            raise RuntimeError("Injected evaluation failure")
        return original_apply(context, obj)
    operators.apply_enabled_modifiers = fail_one
    try:
        assert bpy.ops.mdt.apply_modifier_keep_shape_keys() == {"FINISHED"}
        assert keyed.modifiers.get("FinalSubdivision") is not None
        assert plain.modifiers.get("FinalSubdivision") is None
    finally:
        operators.apply_enabled_modifiers = original_apply
    print("MODIFIER_BATCH_SMOKE_OK: mixed meshes, shape mix, disabled/unselected/shared data, edit mode, partial failure")


if __name__ == "__main__":
    run()
