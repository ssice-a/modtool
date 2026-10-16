"""Target-preserving weld for two selected boundary loops on one mesh."""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import bmesh
import numpy as np
from mathutils.kdtree import KDTree

from .loop_selection import (
    active_edge_for_component,
    edit_mesh_objects,
    order_edge_cycle,
    selected_edge_components,
)


_EPSILON = 1.0e-8
_AREA_EPSILON = 1.0e-12
_MAX_MEAN_SEAM_GAP_RATIO = 3.0
_MAX_SEAM_GAP_RATIO = 8.0
_ROLE_SOURCE = 1
_ROLE_TARGET = 2


@dataclass(frozen=True)
class _SelectedCycle:
    obj: object
    vertex_indices: tuple[int, ...]
    edge_vertex_indices: tuple[tuple[int, int], ...]
    active_edge: tuple[int, int] | None = None


def _analyze_selection(context):
    objects = edit_mesh_objects(context)
    if len(objects) != 1:
        raise ValueError(
            "Bridge and Weld Loops requires one mesh in Edit Mode. Join the two parts first."
        )
    obj = objects[0]
    if context.view_layer.objects.active != obj:
        raise ValueError("The active mesh must contain both selected boundary loops.")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    components = selected_edge_components(bm, boundary_only=True)
    if len(components) != 2:
        raise ValueError("Select exactly two closed boundary loops.")

    active_element = bm.select_history.active
    active_index = None
    records = []
    for index, component in enumerate(components):
        active_edge = active_edge_for_component(component, active_element)
        verts, _edges = order_edge_cycle(component, active_edge)
        records.append(
            _SelectedCycle(
                obj=obj,
                vertex_indices=tuple(vert.index for vert in verts),
                edge_vertex_indices=tuple(
                    (verts[i].index, verts[(i + 1) % len(verts)].index)
                    for i in range(len(verts))
                ),
                active_edge=(active_edge.verts[0].index, active_edge.verts[1].index)
                if active_edge is not None
                else None,
            )
        )
        if active_edge is not None:
            active_index = index

    if active_index is None:
        raise ValueError("Make an edge or vertex in the target loop active.")
    target = records[active_index]
    source = records[1 - active_index]
    return source, target


def _new_int_layer(layer_collection, base_name):
    layer = layer_collection.get(base_name)
    if layer is not None:
        layer_collection.remove(layer)
    return layer_collection.new(base_name)


def _tag_cycle_edges(bm, cycle, role_layer, role):
    for first_index, second_index in cycle.edge_vertex_indices:
        edge = bm.edges.get((bm.verts[first_index], bm.verts[second_index]))
        if edge is None or len(edge.link_faces) != 1:
            raise ValueError("A selected boundary changed before bridge preparation.")
        edge[role_layer] = role


def _boundary_edges_for_role(bm, role_layer, role):
    return [
        edge
        for edge in bm.edges
        if len(edge.link_faces) == 1 and int(edge[role_layer]) == role
    ]


def _ordered_tagged_cycle(bm, role_layer, role):
    edges = _boundary_edges_for_role(bm, role_layer, role)
    if len(edges) < 3:
        raise ValueError("The selected loop changed while inserting matching vertices.")
    adjacency = {}
    for edge in edges:
        for vert in edge.verts:
            adjacency.setdefault(vert, []).append(edge)
    if len(adjacency) != len(edges) or any(len(item) != 2 for item in adjacency.values()):
        raise ValueError("A selected boundary is no longer a simple closed loop.")
    first_edge = edges[0]
    first, second = first_edge.verts
    ordered_verts = [first, second]
    ordered_edges = [first_edge]
    previous, current = first_edge, second
    while len(ordered_edges) < len(edges):
        next_edge = next(edge for edge in adjacency[current] if edge != previous)
        next_vert = next_edge.other_vert(current)
        ordered_edges.append(next_edge)
        if next_vert == first:
            break
        ordered_verts.append(next_vert)
        previous, current = next_edge, next_vert
    if len(ordered_verts) != len(edges):
        raise ValueError("The selected loop could not be ordered after matching vertices were inserted.")
    return ordered_verts, ordered_edges


def _insert_one_vertex(bm, edge, role_layer, role, inserted_layer):
    result = bmesh.ops.subdivide_edges(
        bm,
        edges=[edge],
        cuts=1,
        use_grid_fill=False,
    )
    created = [
        geom for geom in result.get("geom_inner", ()) if isinstance(geom, bmesh.types.BMVert)
    ]
    if len(created) != 1:
        raise RuntimeError("Blender did not create exactly one matching boundary vertex.")
    created[0][inserted_layer] = 1
    for geometry in result.get("geom_split", ()):
        if isinstance(geometry, bmesh.types.BMEdge):
            geometry[role_layer] = role


def _equalize_counts(bm, role_layer, inserted_layer, source_count, target_count):
    if source_count == target_count:
        return
    role = _ROLE_SOURCE if source_count < target_count else _ROLE_TARGET
    for _ in range(abs(source_count - target_count)):
        edges = _boundary_edges_for_role(bm, role_layer, role)
        if not edges:
            raise ValueError("Could not find an edge on the shorter boundary for insertion.")
        edge = max(edges, key=lambda item: (item.verts[0].co - item.verts[1].co).length_squared)
        _insert_one_vertex(bm, edge, role_layer, role, inserted_layer)


def _coordinate_array(verts):
    values = np.empty((len(verts), 3), dtype=np.float64)
    for index, vert in enumerate(verts):
        values[index] = vert.co
    return values


def _best_correspondence(source_verts, target_verts):
    source = _coordinate_array(source_verts)
    target = _coordinate_array(target_verts)
    if len(source) != len(target):
        raise RuntimeError("Boundary counts were not equalized before correspondence.")
    source_norm = float(np.square(source).sum())
    target_norm = float(np.square(target).sum())

    def best_offset(candidate):
        # Circular cross-correlation scores every cyclic phase in O(n log n),
        # avoiding the former O(n^2) loop comparison on dense boundaries.
        correlation = sum(
            np.fft.ifft(
                np.conj(np.fft.fft(source[:, axis]))
                * np.fft.fft(candidate[:, axis])
            ).real
            for axis in range(3)
        )
        squared_costs = source_norm + target_norm - (2.0 * correlation)
        offset = int(np.argmin(squared_costs))
        indices = np.roll(np.arange(len(candidate), dtype=np.int32), -offset)
        distances = np.linalg.norm(source - candidate[indices], axis=1)
        return (
            max(0.0, float(squared_costs[offset])),
            float(distances.max(initial=0.0)),
            indices,
        )

    candidates = []
    for candidate, reversed_order in (
        (target, False),
        (np.concatenate((target[:1], target[:0:-1]), axis=0), True),
    ):
        cost, maximum, indices = best_offset(candidate)
        candidates.append((cost, maximum, indices, reversed_order))
    best = min(candidates, key=lambda item: (item[0], item[1]))
    distance = float(np.sqrt(best[0] / max(len(source), 1)))
    # Recover target-vertex references from coordinates by retaining index
    # order alongside the candidate orientation.
    base_indices = np.arange(len(target_verts), dtype=np.int32)
    if best[3]:
        base_indices = np.concatenate((base_indices[:1], base_indices[:0:-1]))
    ordered_indices = base_indices[best[2]]
    return [target_verts[int(index)] for index in ordered_indices], distance, best[1]


def _loop_reference_edge_length(source_verts, target_verts):
    lengths = []
    for verts in (source_verts, target_verts):
        coordinates = _coordinate_array(verts)
        lengths.extend(np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1))
    return max(float(np.median(np.asarray(lengths, dtype=np.float64))), _EPSILON)


def _prepare_cycles(bm, source, target, role_layer, inserted_layer):
    """Insert only on the shorter loop and return its final one-to-one mapping."""
    _equalize_counts(
        bm,
        role_layer,
        inserted_layer,
        len(source.vertex_indices),
        len(target.vertex_indices),
    )
    source_verts, _ = _ordered_tagged_cycle(bm, role_layer, _ROLE_SOURCE)
    target_verts, _ = _ordered_tagged_cycle(bm, role_layer, _ROLE_TARGET)
    assignment, mean_distance, max_distance = _best_correspondence(source_verts, target_verts)
    reference_edge = _loop_reference_edge_length(source_verts, target_verts)
    return (
        source_verts,
        target_verts,
        assignment,
        mean_distance,
        max_distance,
        reference_edge,
    )


def _validate_seam_quality(mean_distance, max_distance, reference_edge):
    mean_ratio = mean_distance / reference_edge
    max_ratio = max_distance / reference_edge
    if mean_ratio > _MAX_MEAN_SEAM_GAP_RATIO or max_ratio > _MAX_SEAM_GAP_RATIO:
        raise ValueError(
            "The selected loops are too far apart for a weld: "
            f"mean gap {mean_distance:.4g} ({mean_ratio:.2f}x edge), "
            f"max gap {max_distance:.4g} ({max_ratio:.2f}x edge). "
            "Move/sculpt the source loop closer, then try again."
        )


def _validate_original_seam(bm, source, target):
    """Reject distant loops before inserting any topology."""
    source_verts = [bm.verts[index] for index in source.vertex_indices]
    target_verts = [bm.verts[index] for index in target.vertex_indices]
    # BMesh is the source of truth in Edit Mode. Read only the selected loop
    # coordinates so this preflight cannot trigger a shape-key mesh sync.
    source_coordinates = np.asarray([tuple(vert.co) for vert in source_verts], dtype=np.float64)
    target_coordinates = np.asarray([tuple(vert.co) for vert in target_verts], dtype=np.float64)
    source_edges = np.linalg.norm(
        np.roll(source_coordinates, -1, axis=0) - source_coordinates,
        axis=1,
    )
    target_edges = np.linalg.norm(
        np.roll(target_coordinates, -1, axis=0) - target_coordinates,
        axis=1,
    )
    reference_edge = max(
        float(np.median(np.concatenate((source_edges, target_edges)))), _EPSILON
    )
    # A nearest-neighbour estimate is intentionally only a rejection guard.
    # Exact cyclic correspondence follows after equalization.
    def nearest_distances(query, reference):
        tree = KDTree(len(reference))
        for index, coordinate in enumerate(reference):
            tree.insert(tuple(coordinate), index)
        tree.balance()
        return np.asarray([tree.find(tuple(coordinate))[2] for coordinate in query])

    source_to_target = nearest_distances(source_coordinates, target_coordinates)
    target_to_source = nearest_distances(target_coordinates, source_coordinates)
    _validate_seam_quality(
        float((source_to_target.mean() + target_to_source.mean()) * 0.5),
        float(max(source_to_target.max(initial=0.0), target_to_source.max(initial=0.0))),
        reference_edge,
    )


def _set_shape_data_to_target(source_verts, target_verts, shape_layers):
    """Make the target authoritative in every shape layer before welding."""
    for source, target in zip(source_verts, target_verts):
        source.co = target.co
        for layer in shape_layers:
            source[layer] = target[layer]


def _remove_invalid_affected_faces(bm, faces):
    invalid = []
    for face in faces:
        if not face.is_valid:
            continue
        if len({vert for vert in face.verts}) < 3 or face.calc_area() <= _AREA_EPSILON:
            invalid.append(face)
    if invalid:
        bmesh.ops.delete(bm, geom=invalid, context="FACES")
    return len(invalid)


def _triangulate_non_quads(bm, faces):
    candidates = [
        face
        for face in faces
        if face.is_valid and len(face.verts) > 4 and face.calc_area() > _AREA_EPSILON
    ]
    if not candidates:
        return 0
    bmesh.ops.triangulate(
        bm,
        faces=candidates,
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    return len(candidates)


def _face_quality(face):
    points = [vert.co.copy() for vert in face.verts]
    lengths = [
        (points[(index + 1) % len(points)] - points[index]).length
        for index in range(len(points))
    ]
    minimum = min(lengths, default=0.0)
    aspect = float("inf") if minimum <= _EPSILON else max(lengths) / minimum
    warp = 0.0
    if len(points) == 4:
        first = (points[1] - points[0]).cross(points[2] - points[0])
        second = (points[2] - points[0]).cross(points[3] - points[0])
        warp = 180.0 if min(first.length, second.length) <= _EPSILON else degrees(first.angle(second))
    return aspect, warp


def _cleanup_temporary_layers(bm, role_layer, inserted_layer):
    if role_layer is not None:
        bm.edges.layers.int.remove(role_layer)
    if inserted_layer is not None:
        bm.verts.layers.int.remove(inserted_layer)


def bridge_selected_loops(context):
    """Weld the non-active boundary loop onto the active target loop.

    The active edge/vertex identifies the target loop. Original target vertices
    are never moved or deleted. A shorter loop receives interpolated vertices;
    source vertices are then welded to the resulting target vertices. No faces
    are created between the loops.
    """
    if context.mode != "EDIT_MESH":
        raise ValueError("Run Bridge and Weld Loops from Mesh Edit Mode.")
    source, target = _analyze_selection(context)
    obj = target.obj
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    role_layer = None
    inserted_layer = None
    welded = False
    try:
        _validate_original_seam(bm, source, target)
        role_layer = _new_int_layer(bm.edges.layers.int, ".mdt_weld_edge_role")
        inserted_layer = _new_int_layer(bm.verts.layers.int, ".mdt_weld_inserted")
        _tag_cycle_edges(bm, source, role_layer, _ROLE_SOURCE)
        _tag_cycle_edges(bm, target, role_layer, _ROLE_TARGET)
        (
            source_verts,
            target_verts,
            assignment,
            mean_distance,
            max_distance,
            reference_edge,
        ) = _prepare_cycles(bm, source, target, role_layer, inserted_layer)
        _validate_seam_quality(mean_distance, max_distance, reference_edge)

        shape_layers = list(bm.verts.layers.shape)
        affected_faces = {
            face
            for vert in source_verts + target_verts
            for face in vert.link_faces
        }
        _set_shape_data_to_target(source_verts, assignment, shape_layers)
        bmesh.ops.weld_verts(
            bm,
            targetmap={source_vert: target_vert for source_vert, target_vert in zip(source_verts, assignment)},
        )
        welded = True
        _cleanup_temporary_layers(bm, role_layer, inserted_layer)
        role_layer = inserted_layer = None
        removed_faces = _remove_invalid_affected_faces(bm, affected_faces)
        triangulated_faces = _triangulate_non_quads(bm, affected_faces)
        bm.normal_update()
        for vert in bm.verts:
            vert.select = False
        for vert in assignment:
            if vert.is_valid:
                vert.select = True
                for edge in vert.link_edges:
                    edge.select = True
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

        max_aspect = 0.0
        max_warp = 0.0
        for face in affected_faces:
            if face.is_valid:
                aspect, warp = _face_quality(face)
                max_aspect = max(max_aspect, aspect)
                max_warp = max(max_warp, warp)
        return {
            "object": obj,
            "source_vertex_count": len(source_verts),
            "target_vertex_count": len(target_verts),
            "inserted_vertex_count": abs(len(source.vertex_indices) - len(target.vertex_indices)),
            "welded_vertex_count": len(source_verts),
            "new_face_count": 0,
            "removed_face_count": removed_faces,
            "triangulated_face_count": triangulated_faces,
            "triangle_count": sum(
                1 for face in affected_faces if face.is_valid and len(face.verts) == 3
            ),
            "mean_seam_distance": mean_distance,
            "max_seam_distance": max_distance,
            "max_face_aspect": max_aspect,
            "max_quad_warp_degrees": max_warp,
        }
    except Exception:
        if inserted_layer is not None and not welded:
            valid_inserted = [vert for vert in bm.verts if int(vert[inserted_layer]) == 1]
            if valid_inserted:
                bmesh.ops.dissolve_verts(bm, verts=valid_inserted, use_face_split=False, use_boundary_tear=False)
        _cleanup_temporary_layers(bm, role_layer, inserted_layer)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        raise
