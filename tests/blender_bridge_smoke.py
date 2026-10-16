"""Background smoke tests for target-preserving loop welding."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from modtool.core import bridge_loops  # noqa: E402


bridge_selected_loops = bridge_loops.bridge_selected_loops


def _reset():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _make_combined(source_count, target_count):
    vertices = []
    faces = []
    for count, radius, inner_z, reverse in (
        (source_count, 1.0, -1.0, False),
        (target_count, 1.0, 1.0, True),
    ):
        base = len(vertices)
        seam_z = 0.0 if len(vertices) == 0 else 0.015
        for z in (seam_z, inner_z):
            vertices.extend(
                (
                    radius * math.cos(2.0 * math.pi * index / count),
                    radius * math.sin(2.0 * math.pi * index / count),
                    z,
                )
                for index in range(count)
            )
        for index in range(count):
            nxt = (index + 1) % count
            face = (base + index, base + nxt, base + count + nxt, base + count + index)
            faces.append(tuple(reversed(face)) if reverse else face)

    mesh = bpy.data.meshes.new("BridgeSmokeMesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update()
    obj = bpy.data.objects.new("BridgeSmoke", mesh)
    bpy.context.collection.objects.link(obj)
    for layer_index in range(2):
        uv = mesh.uv_layers.new(name=f"UV{layer_index}")
        for loop in mesh.loops:
            vertex = mesh.vertices[loop.vertex_index].co
            uv.data[loop.index].uv = (vertex.x * 0.25 + layer_index, vertex.y * 0.25)
    shape = obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="VisibleShape")
    for point in key.data:
        point.co.z += 0.03 * point.co.x
    key.value = 0.5
    left = obj.vertex_groups.new(name="Left")
    right = obj.vertex_groups.new(name="Right")
    for index, vertex in enumerate(mesh.vertices):
        left.add((index,), max(0.0, min(1.0, 0.5 + vertex.co.x * 0.25)), "REPLACE")
        right.add((index,), max(0.0, min(1.0, 0.5 - vertex.co.x * 0.25)), "REPLACE")
    return obj


def _select_loops(obj, source_count, target_count):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    for edge in bm.edges:
        edge.select = False
    source_edges = [
        edge
        for edge in bm.edges
        if len(edge.link_faces) == 1
        and all(vertex.index < source_count for vertex in edge.verts)
    ]
    target_start = source_count * 2
    target_edges = [
        edge
        for edge in bm.edges
        if len(edge.link_faces) == 1
        and all(target_start <= vertex.index < target_start + target_count for vertex in edge.verts)
    ]
    assert len(source_edges) == source_count
    assert len(target_edges) == target_count
    for edge in source_edges + target_edges:
        edge.select = True
    bm.select_history.clear()
    bm.select_history.add(target_edges[0])
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def _assert_mesh_valid(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    assert not mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()
    assert all(len(set(polygon.vertices)) >= 3 for polygon in mesh.polygons)
    assert all(math.isfinite(value) for vertex in mesh.vertices for value in vertex.co)
    assert len(mesh.uv_layers) == 2
    assert all(
        math.isfinite(value)
        for uv_layer in mesh.uv_layers
        for item in uv_layer.data
        for value in item.uv
    )
    assert mesh.shape_keys is not None
    assert [key.name for key in mesh.shape_keys.key_blocks] == ["Basis", "VisibleShape"]
    assert abs(mesh.shape_keys.key_blocks["VisibleShape"].value - 0.5) < 1.0e-6
    assert all(len(key.data) == len(mesh.vertices) for key in mesh.shape_keys.key_blocks)
    assert all(
        math.isfinite(value)
        for polygon in mesh.polygons
        for value in polygon.normal
    )


def _run_case(source_count, target_count):
    _reset()
    obj = _make_combined(source_count, target_count)
    target_original = [
        tuple(obj.data.vertices[index].co)
        for index in range(source_count * 2, source_count * 2 + target_count)
    ]
    original_face_count = len(obj.data.polygons)
    _select_loops(obj, source_count, target_count)
    result = bridge_selected_loops(bpy.context)
    assert result["object"] == obj
    assert result["new_face_count"] == 0
    assert result["inserted_vertex_count"] == abs(source_count - target_count)
    assert result["welded_vertex_count"] == max(source_count, target_count)
    assert len(obj.data.polygons) >= original_face_count
    bpy.ops.object.mode_set(mode="OBJECT")
    assert len(obj.data.vertices) == (
        source_count * 2
        + target_count * 2
        + abs(source_count - target_count)
        - max(source_count, target_count)
    )
    remaining_target = [
        tuple(vertex.co)
        for vertex in obj.data.vertices
        if abs(vertex.co.z - 0.015) < 1.0e-7
    ]
    assert all(any(sum((a[i] - b[i]) ** 2 for i in range(3)) < 1.0e-12 for b in remaining_target) for a in target_original)
    _assert_mesh_valid(obj)
    print(
        f"BRIDGE_WELD_OK {source_count}->{target_count}: "
        f"{result['inserted_vertex_count']} inserted, "
        f"{result['welded_vertex_count']} welded, no bridge faces"
    )


def _run_rejection_case():
    _reset()
    source_count, target_count = 8, 12
    obj = _make_combined(source_count, target_count)
    _select_loops(obj, source_count, target_count)
    bm = bmesh.from_edit_mesh(obj.data)
    for vertex in bm.verts:
        if vertex.index < source_count * 2:
            vertex.co.z -= 20.0
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    before = (len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons))
    try:
        bridge_selected_loops(bpy.context)
    except ValueError as exc:
        assert "too far apart" in str(exc)
    else:
        raise AssertionError("Distant loops must be rejected before topology changes.")
    bm = bmesh.from_edit_mesh(obj.data)
    after = (len(bm.verts), len(bm.edges), len(bm.faces))
    assert after == before
    assert bm.verts.layers.int.get(".mdt_weld_role") is None
    assert bm.verts.layers.int.get(".mdt_weld_anchor") is None
    print("BRIDGE_WELD_REJECTION_OK")


def _run_insert_rollback_case():
    _reset()
    source_count, target_count = 8, 12
    obj = _make_combined(source_count, target_count)
    _select_loops(obj, source_count, target_count)
    bm = bmesh.from_edit_mesh(obj.data)
    before = (len(bm.verts), len(bm.edges), len(bm.faces))
    original = bridge_loops._best_correspondence
    bridge_loops._best_correspondence = lambda *_args: (_ for _ in ()).throw(
        ValueError("forced post-insert failure")
    )
    try:
        try:
            bridge_selected_loops(bpy.context)
        except ValueError as exc:
            assert "forced post-insert failure" in str(exc)
        else:
            raise AssertionError("The forced post-insert failure must propagate.")
    finally:
        bridge_loops._best_correspondence = original
    bm = bmesh.from_edit_mesh(obj.data)
    after = (len(bm.verts), len(bm.edges), len(bm.faces))
    assert after == before
    assert bm.edges.layers.int.get(".mdt_weld_edge_role") is None
    assert bm.verts.layers.int.get(".mdt_weld_inserted") is None
    print("BRIDGE_WELD_INSERT_ROLLBACK_OK")


for _case in ((8, 8), (12, 8), (8, 12)):
    _run_case(*_case)
_run_rejection_case()
_run_insert_rollback_case()
print("BLENDER_BRIDGE_WELD_SMOKE_OK")
