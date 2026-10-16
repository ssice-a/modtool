"""Edit Mode regression tests for selected-loop seam weight transfer."""

import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modtool.core.seam import unify_selected_loop_weights, _loop_weight_samples
from modtool.ops.group_mapping import MDT_OT_unify_seam_weights_from_target


SQUARE = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
DENSE = [(0, 0, 0), (0.5, 0, 0.02), (2, 0, 0), (2, 1, 0.02),
         (2, 2, 0), (1, 2, 0.02), (0, 2, 0), (0, 1, 0.02)]
REFERENCE = [{"A": 0.2, "Tiny": 1.0e-10}, {"B": 0.8}, {"A": 0.6}, {"B": 0.4}]


def reset():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def mesh_object(name, rings, weighted=True):
    vertices, faces, ring_ids = [], [], []
    for points in rings:
        start, count = len(vertices), len(points)
        vertices.extend(points)
        vertices.extend((x, y, z + 1) for x, y, z in points)
        ring_ids.append(list(range(start, start + count)))
        for i in range(count):
            j = (i + 1) % count
            faces.append((start + i, start + j, start + j + count, start + i + count))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    uv = mesh.uv_layers.new(name="UVMap")
    values = np.arange(len(mesh.loops) * 2, dtype=np.float32) * 0.01
    uv.data.foreach_set("uv", values)
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Shape")
    for point in key.data:
        point.co.z += 0.1
    key.value = 0.6
    if weighted:
        obj.vertex_groups.new(name="Old").add(list(range(len(vertices))), 0.7, "REPLACE")
    return obj, ring_ids


def assign_reference(obj, indices):
    # Deliberately choose group order unrelated to alphabetical order.
    for name in ("B", "Tiny", "A"):
        obj.vertex_groups.new(name=name)
    for index, weights in zip(indices, REFERENCE):
        obj.vertex_groups["Old"].remove([index])
        for name, value in weights.items():
            obj.vertex_groups[name].add([index], value, "REPLACE")


def enter_selection(records, reference_obj, reference_indices, vertex_active=False):
    bpy.ops.object.select_all(action="DESELECT")
    for obj, _rings in records:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = reference_obj
    bpy.context.tool_settings.mesh_select_mode = (vertex_active, not vertex_active, False)
    bpy.ops.object.mode_set(mode="EDIT")
    for obj, rings in records:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for face in bm.faces:
            face.select_set(False)
        for edge in bm.edges:
            edge.select_set(False)
        for vert in bm.verts:
            vert.select_set(False)
        bm.select_history.clear()
        for indices in rings:
            for i, first in enumerate(indices):
                second = indices[(i + 1) % len(indices)]
                bm.edges.get((bm.verts[first], bm.verts[second])).select_set(True)
        if obj == reference_obj:
            element = (bm.verts[reference_indices[0]] if vertex_active else
                       bm.edges.get((bm.verts[reference_indices[0]], bm.verts[reference_indices[1]])))
            bm.select_history.add(element)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def weights(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    names = {g.index: g.name for g in obj.vertex_groups}
    layer = bm.verts.layers.deform.active
    if layer is None:
        return [{} for _vert in bm.verts]
    return [{names[i]: value for i, value in vert[layer].items()} for vert in bm.verts]


def unchanged_data(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    uv = bm.loops.layers.uv.active
    shapes = list(bm.verts.layers.shape)
    active = bm.select_history.active
    return (
        [(tuple(v.co), v.select, [tuple(v[s]) for s in shapes]) for v in bm.verts],
        [(tuple(v.index for v in e.verts), e.select) for e in bm.edges],
        [(tuple(v.index for v in f.verts), f.select, [tuple(l[uv].uv) for l in f.loops]) for f in bm.faces],
        (type(active).__name__, active.index) if active is not None else None,
        [(k.name, k.value) for k in obj.data.shape_keys.key_blocks],
    )


def assert_weights(actual, expected):
    assert actual.keys() == expected.keys(), (actual, expected)
    for name in expected:
        assert abs(actual[name] - expected[name]) < max(1.0e-15, abs(expected[name]) * 1.0e-6), (actual, expected)


def single_mesh_case():
    reset()
    obj, (dest, ref) = mesh_object("Combined", [DENSE, SQUARE])
    assign_reference(obj, ref)
    enter_selection([(obj, [dest, ref])], obj, ref, vertex_active=True)
    before, before_weights = unchanged_data(obj), weights(obj)
    result = unify_selected_loop_weights(bpy.context)
    assert result["copied_vertex_count"] == 4 and result["interpolated_vertex_count"] == 4, result
    expected = [REFERENCE[0], {"A": 0.15, "Tiny": 7.5e-11, "B": 0.2}, REFERENCE[1],
                {"B": 0.4, "A": 0.3}, REFERENCE[2], {"A": 0.3, "B": 0.2},
                REFERENCE[3], {"B": 0.2, "A": 0.1, "Tiny": 5.0e-11}]
    after = weights(obj)
    for index, wanted in zip(dest, expected):
        assert_weights(after[index], wanted)
    for index in set(range(len(after))) - set(dest):
        assert after[index] == before_weights[index]
    after_data = unchanged_data(obj)
    assert after_data == before, [(i, a, b) for i, (a, b) in enumerate(zip(before, after_data)) if a != b]
    assert bpy.context.mode == "EDIT_MESH"
    assert bpy.ops.mdt.unify_seam_weights_from_target() == {"FINISHED"}
    assert weights(obj) == after
    print("SEAM_SAME_MESH_UNEQUAL_COUNTS_SHAPE_UV_SELECTION_OK")


def multi_mesh_case(empty_destination=False):
    reset()
    reference, (ref,) = mesh_object("Reference", [SQUARE])
    assign_reference(reference, ref)
    # Geometry is identical in world space, with reversed winding and differing transforms.
    dest_points = [(x / 2 - 3, y, z) for x, y, z in reversed(SQUARE)]
    dest_obj, (dest,) = mesh_object("Destination", [dest_points], weighted=not empty_destination)
    dest_obj.matrix_world = Matrix.Translation((6, 0, 0)) @ Matrix.Diagonal((2, 1, 1, 1))
    enter_selection([(dest_obj, [dest]), (reference, [ref])], reference, ref)
    before_ref, before_dest = weights(reference), weights(dest_obj)
    before = unchanged_data(dest_obj)
    # Sole reference loop on the active object remains unambiguous without selection history.
    bmesh.from_edit_mesh(reference.data).select_history.clear()
    assert bpy.ops.mdt.unify_seam_weights_from_target() == {"FINISHED"}
    after = weights(dest_obj)
    for index, wanted in zip(dest, reversed(REFERENCE)):
        assert_weights(after[index], wanted)
    for index in set(range(len(after))) - set(dest):
        assert before_dest[index] == after[index]
    assert weights(reference) == before_ref
    assert unchanged_data(dest_obj) == before
    assert bpy.context.mode == "EDIT_MESH"
    print(f"SEAM_MULTI_OBJECT_WORLD_SPACE_GROUP_NAMES_OK empty={empty_destination}")


def reject_case(kind):
    reset()
    obj, (dest, ref) = mesh_object("Invalid", [DENSE, SQUARE])
    assign_reference(obj, ref)
    enter_selection([(obj, [dest, ref])], obj, ref)
    bm = bmesh.from_edit_mesh(obj.data)
    if kind == "open":
        bm.edges.get((bm.verts[dest[0]], bm.verts[dest[1]])).select_set(False)
    elif kind == "active":
        bm.select_history.clear()
    elif kind == "empty":
        layer = bm.verts.layers.deform.active
        for index in ref:
            bm.verts[index][layer].clear()
    before, before_weights = unchanged_data(obj), weights(obj)
    try:
        unify_selected_loop_weights(bpy.context)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid selection was accepted: {kind}")
    assert unchanged_data(obj) == before and weights(obj) == before_weights
    assert bpy.context.mode == "EDIT_MESH"
    print(f"SEAM_REJECTION_UNCHANGED_OK {kind}")


def segment_search_case():
    # A long nearest segment can have distant endpoints. Check against exhaustive projections.
    target = np.asarray([(-100, 0, 0), (100, 0, 0), (1, 1, 0), (-1, 1, 0)], dtype=float)
    rng = np.random.default_rng(24)
    source = rng.uniform((-30, -1, -0.1), (30, 2, 0.1), (100, 3))
    samples, _count, _maximum = _loop_weight_samples(source, target, 0.0)
    directions = np.roll(target, -1, axis=0) - target
    for point, (first, second, factor) in zip(source, samples):
        t = np.clip(np.sum((point - target) * directions, axis=1) / np.sum(directions ** 2, axis=1), 0, 1)
        expected = np.min(np.sum((point - target - t[:, None] * directions) ** 2, axis=1))
        found = np.sum((point - ((1 - factor) * target[first] + factor * target[second])) ** 2)
        assert abs(found - expected) < 1.0e-10
    # Duplicate positions produce zero-length edges, without NaN or infinite weights.
    duplicate = np.insert(target, 1, target[0], axis=0)
    samples, _count, _maximum = _loop_weight_samples(source[:3], duplicate, 0)
    assert all(np.isfinite(factor) for _a, _b, factor in samples)
    print("SEAM_SEGMENT_SEARCH_LONG_AND_DEGENERATE_EDGES_OK")


bpy.utils.register_class(MDT_OT_unify_seam_weights_from_target)
single_mesh_case()
multi_mesh_case()
multi_mesh_case(empty_destination=True)
for kind in ("open", "active", "empty"):
    reject_case(kind)
segment_search_case()
print("BLENDER_SEAM_WEIGHTS_SMOKE_OK")
