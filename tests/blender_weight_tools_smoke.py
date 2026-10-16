"""Blender background regression tests for weight mirror and island cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

sys.dont_write_bytecode = True


REPOSITORY_PARENT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_PARENT not in sys.path:
    sys.path.insert(0, REPOSITORY_PARENT)

from modtool.core.rename import (  # noqa: E402
    clean_disconnected_vertex_group_weights,
    mirror_vertex_group_weights_x,
)


def _reset_scene():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _create_armature(bone_names):
    armature_data = bpy.data.armatures.new("WeightTestArmature")
    armature = bpy.data.objects.new("WeightTestArmature", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for index, name in enumerate(bone_names):
        bone = armature_data.edit_bones.new(name)
        bone.head = (0.0, 0.0, float(index))
        bone.tail = (0.0, 0.0, float(index + 1))
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _create_mesh(name, vertices, edges, bone_names, faces=()):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    armature = _create_armature(bone_names)
    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    for bone_name in bone_names:
        obj.vertex_groups.new(name=bone_name)
    return obj


def _weight(obj, group_name, vertex_index):
    try:
        return obj.vertex_groups[group_name].weight(vertex_index)
    except RuntimeError:
        return 0.0


def _deform_sum(obj, vertex_index, group_names):
    return sum(_weight(obj, name, vertex_index) for name in group_names)


def _test_mirror_copies_complete_deform_vectors():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "MirrorWeights",
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        (),
        names,
        ((0, 2, 3, 1),),
    )
    obj.vertex_groups["Bone-R-Test"].add((0,), 0.6, "REPLACE")
    obj.vertex_groups["Center"].add((0,), 0.4, "REPLACE")
    obj.vertex_groups["Bone-L-Test"].add((1,), 0.2, "REPLACE")
    obj.vertex_groups["Center"].add((1,), 0.8, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((2,), 0.25, "REPLACE")
    obj.vertex_groups["Center"].add((2,), 0.75, "REPLACE")
    obj.vertex_groups["Bone-L-Test"].add((3,), 0.9, "REPLACE")
    obj.vertex_groups["Center"].add((3,), 0.1, "REPLACE")

    mirror_vertex_group_weights_x(obj, direction="RIGHT_TO_LEFT")

    assert abs(_weight(obj, "Bone-L-Test", 1) - 0.6) < 1.0e-6
    assert abs(_weight(obj, "Center", 1) - 0.4) < 1.0e-6
    assert abs(_weight(obj, "Bone-L-Test", 3) - 0.25) < 1.0e-6
    assert abs(_weight(obj, "Center", 3) - 0.75) < 1.0e-6
    assert abs(_deform_sum(obj, 1, names) - 1.0) < 1.0e-6
    assert abs(_deform_sum(obj, 3, names) - 1.0) < 1.0e-6
    print("MIRROR_COMPLETE_VECTOR_OK")


def _test_mirror_interpolates_different_side_topology():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "MirrorDifferentTopology",
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, 0.5, 0.5),
        ),
        (),
        names,
        ((0, 1, 2, 3), (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 4, 8)),
    )
    left_weights = (0.2, 0.8, 0.6, 0.4)
    for vertex_index, weight in enumerate(left_weights):
        obj.vertex_groups["Bone-L-Test"].add((vertex_index,), weight, "REPLACE")
        obj.vertex_groups["Center"].add((vertex_index,), 1.0 - weight, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((4, 5, 6, 7, 8), 1.0, "REPLACE")

    result = mirror_vertex_group_weights_x(obj, direction="LEFT_TO_RIGHT")

    for source_index, target_index in ((0, 4), (1, 5), (2, 6), (3, 7)):
        expected = left_weights[source_index]
        assert abs(_weight(obj, "Bone-R-Test", target_index) - expected) < 1.0e-6
        assert abs(_weight(obj, "Center", target_index) - (1.0 - expected)) < 1.0e-6
    center_side_weight = _weight(obj, "Bone-R-Test", 8)
    center_weight = _weight(obj, "Center", 8)
    assert abs(center_side_weight - 0.4) < 1.0e-6, center_side_weight
    assert abs(center_weight - 0.6) < 1.0e-6, center_weight
    assert result["unmatched_vertex_count"] == 0
    print("MIRROR_SURFACE_INTERPOLATION_OK")


def _test_mirror_auto_tolerance_uses_local_surface_scale():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    vertices = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-1.6, 0.0, 0.0),
        (-1.6, 1.0, 0.0),
        (-1.6, 1.0, 1.0),
        (-1.6, 0.0, 1.0),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    for index in range(8):
        y = 10.0 + index
        start = len(vertices)
        vertices.extend(((0.0, y, 0.0), (0.0, y + 0.1, 0.0), (0.0, y, 0.1)))
        faces.append((start, start + 1, start + 2))

    obj = _create_mesh("MirrorLocalScale", vertices, (), names, faces)
    obj.vertex_groups["Bone-L-Test"].add((0, 1, 2, 3), 0.75, "REPLACE")
    obj.vertex_groups["Center"].add((0, 1, 2, 3), 0.25, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((4, 5, 6, 7), 1.0, "REPLACE")

    result = mirror_vertex_group_weights_x(obj, direction="LEFT_TO_RIGHT")

    for vertex_index in (4, 5, 6, 7):
        assert abs(_weight(obj, "Bone-R-Test", vertex_index) - 0.75) < 1.0e-6
        assert abs(_weight(obj, "Center", vertex_index) - 0.25) < 1.0e-6
    assert result["unmatched_vertex_count"] == 0
    print("MIRROR_LOCAL_SCALE_TOLERANCE_OK")


def _test_island_cleanup_normalizes_and_preserves_last_weight():
    _reset_scene()
    names = ("Bone-L-Test", "Center")
    obj = _create_mesh(
        "CleanWeights",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        ((0, 1), (2, 3)),
        names,
    )
    obj.vertex_groups["Bone-L-Test"].add((0, 1), 0.8, "REPLACE")
    obj.vertex_groups["Center"].add((0, 1), 0.2, "REPLACE")
    obj.vertex_groups["Bone-L-Test"].add((2,), 0.25, "REPLACE")
    obj.vertex_groups["Center"].add((2,), 0.75, "REPLACE")
    obj.vertex_groups["Bone-L-Test"].add((3,), 1.0, "REPLACE")

    result = clean_disconnected_vertex_group_weights(obj)

    assert _weight(obj, "Bone-L-Test", 2) == 0.0
    assert abs(_weight(obj, "Center", 2) - 1.0) < 1.0e-6
    assert abs(_weight(obj, "Bone-L-Test", 3) - 1.0) < 1.0e-6
    assert abs(_deform_sum(obj, 2, names) - 1.0) < 1.0e-6
    assert abs(_deform_sum(obj, 3, names) - 1.0) < 1.0e-6
    assert result["preserved_last_weight_count"] == 1
    print("CLEAN_ISLAND_NORMALIZATION_OK")


def _test_mirror_compact_twist_names_both_directions():
    for direction, source, target in (
        ("LEFT_TO_RIGHT", 0, 3), ("RIGHT_TO_LEFT", 3, 0),
    ):
        _reset_scene()
        names = ("Bip001_LUpArmTwist1", "Bip001_RUpArmTwist1", "Center")
        obj = _create_mesh(
            "CompactTwist", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                             (-1, 0, 0), (-1, 1, 0), (-1, 0, 1)),
            (), names, ((0, 1, 2), (3, 4, 5)),
        )
        src_name = names[0 if source == 0 else 1]
        dst_name = names[1 if source == 0 else 0]
        obj.vertex_groups[src_name].add(tuple(range(source, source + 3)), .65, "REPLACE")
        obj.vertex_groups["Center"].add(tuple(range(source, source + 3)), .35, "REPLACE")
        obj.vertex_groups[dst_name].add(tuple(range(target, target + 3)), 1, "REPLACE")
        mirror_vertex_group_weights_x(obj, direction)
        assert abs(_weight(obj, dst_name, target) - .65) < 1e-6
        assert _weight(obj, src_name, target) == 0
        assert abs(_weight(obj, src_name, source) - .65) < 1e-6
    print("MIRROR_COMPACT_TWIST_BOTH_DIRECTIONS_OK")


def _test_mirror_preserves_source_and_clears_zero_regions():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "DirectionalMirror", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                              (-1, 0, 0), (-1, 1, 0), (-1, 0, 1)),
        (), names, ((0, 1, 2), (3, 4, 5)),
    )
    obj.vertex_groups["Bone-L-Test"].add((0,), .6, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((0,), .1, "REPLACE")
    obj.vertex_groups["Center"].add((0,), .3, "REPLACE")
    obj.vertex_groups["Center"].add((1, 2), 1, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((3, 4, 5), .9, "REPLACE")
    obj.vertex_groups["Bone-L-Test"].add((3, 4, 5), .05, "REPLACE")
    obj.vertex_groups["Center"].add((3, 4, 5), .05, "REPLACE")
    before = [[_weight(obj, n, i) for n in names] for i in range(3)]
    mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT")
    after = [[_weight(obj, n, i) for n in names] for i in range(3)]
    assert before == after, (before, after)
    assert abs(_weight(obj, "Bone-R-Test", 3) - .6) < 1e-6
    assert abs(_weight(obj, "Bone-L-Test", 3) - .1) < 1e-6
    for i in (4, 5):
        assert _weight(obj, "Bone-R-Test", i) == 0
        assert _weight(obj, "Bone-L-Test", i) == 0
        assert _weight(obj, "Center", i) == 1
    print("MIRROR_SOURCE_AND_ZERO_REGIONS_OK")


def _test_mirror_does_not_implicitly_normalize():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "RawWeightMirror", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                            (-1, 0, 0), (-1, 1, 0), (-1, 0, 1)),
        (), names, ((0, 1, 2), (3, 4, 5)),
    )
    obj.vertex_groups["Bone-L-Test"].add((0, 1, 2), .4, "REPLACE")
    obj.vertex_groups["Center"].add((0, 1, 2), .3, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((3, 4, 5), 1, "REPLACE")
    mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT")
    assert _weight(obj, "Bone-R-Test", 3) == _weight(obj, "Bone-L-Test", 0)
    assert _weight(obj, "Center", 3) == _weight(obj, "Center", 0)
    print("MIRROR_RAW_SOURCE_VALUES_OK")


def _test_mirror_uses_current_shape_and_only_source_surface():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "ShapeAwareMirror", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                             (-.8, .2, 0), (-.8, 1.2, 0), (-.8, .2, 1)),
        (), names, ((0, 1, 2), (3, 4, 5)),
    )
    basis = obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Alignment")
    for i in range(3):
        key.data[i + 3].co = (-1, key.data[i].co.y, key.data[i].co.z)
    key.value = 1
    for i, value in enumerate((.2, .8, .6)):
        obj.vertex_groups["Bone-L-Test"].add((i,), value, "REPLACE")
        obj.vertex_groups["Center"].add((i,), 1 - value, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((3, 4, 5), 1, "REPLACE")
    coords = [tuple(v.co) for v in basis.data]
    result = mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT")
    assert result["exact_vertex_count"] == 3
    assert abs(_weight(obj, "Bone-R-Test", 3) - .2) < 1e-6
    assert [tuple(v.co) for v in basis.data] == coords
    assert key.value == 1

    _reset_scene()
    obj = _create_mesh(
        "SourceSurfaceOnly", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                              (-.2, 0, 0), (-.2, 1, 0), (-.2, 0, 1)),
        (), names, ((0, 1, 2), (3, 4, 5)),
    )
    obj.vertex_groups["Bone-L-Test"].add((0, 1, 2), .6, "REPLACE")
    obj.vertex_groups["Center"].add((0, 1, 2), .4, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((3, 4, 5), 1, "REPLACE")
    mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT", tolerance=1)
    assert abs(_weight(obj, "Bone-R-Test", 3) - .6) < 1e-6
    assert _weight(obj, "Bone-L-Test", 3) == 0
    print("MIRROR_SHAPE_AND_SOURCE_SURFACE_OK")


def _test_mirror_keeps_weight_paint_mode_and_fails_before_locked_write():
    _reset_scene()
    names = ("Bone-L-Test", "Bone-R-Test", "Center")
    obj = _create_mesh(
        "PaintModeMirror", ((1, 0, 0), (1, 1, 0), (1, 0, 1),
                            (-1, 0, 0), (-1, 1, 0), (-1, 0, 1)),
        (), names, ((0, 1, 2), (3, 4, 5)),
    )
    obj.vertex_groups["Bone-L-Test"].add((0, 1, 2), .6, "REPLACE")
    obj.vertex_groups["Center"].add((0, 1, 2), .4, "REPLACE")
    obj.vertex_groups["Bone-R-Test"].add((3, 4, 5), 1, "REPLACE")
    before = [[_weight(obj, n, i) for n in names] for i in range(6)]
    obj.vertex_groups["Center"].lock_weight = True
    try:
        mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT")
    except ValueError as error:
        assert "Unlock" in str(error)
    else:
        raise AssertionError("Locked group was overwritten")
    assert [[_weight(obj, n, i) for n in names] for i in range(6)] == before
    obj.vertex_groups["Center"].lock_weight = False
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    mirror_vertex_group_weights_x(obj, "LEFT_TO_RIGHT")
    assert bpy.context.mode == "PAINT_WEIGHT"
    bpy.ops.object.mode_set(mode="OBJECT")
    print("MIRROR_LOCK_AND_PAINT_MODE_OK")


if __name__ == "__main__":
    _test_mirror_compact_twist_names_both_directions()
    _test_mirror_preserves_source_and_clears_zero_regions()
    _test_mirror_does_not_implicitly_normalize()
    _test_mirror_uses_current_shape_and_only_source_surface()
    _test_mirror_keeps_weight_paint_mode_and_fails_before_locked_write()
    _test_mirror_copies_complete_deform_vectors()
    _test_mirror_interpolates_different_side_topology()
    _test_mirror_auto_tolerance_uses_local_surface_scale()
    _test_island_cleanup_normalizes_and_preserves_last_weight()
    print("WEIGHT_TOOLS_SMOKE_OK")
