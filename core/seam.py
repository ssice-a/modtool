"""Seam vertex-group renaming and selected-loop weight transfer."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import bmesh
import numpy as np
from mathutils.kdtree import KDTree

from .common import is_mesh_object, sync_edit_mesh_data
from .loop_selection import (
    active_edge_for_component,
    edit_mesh_objects,
    order_edge_cycle,
    selected_edge_components,
)
from .rename import apply_exact_name_pairs


DEFAULT_MATCH_TOLERANCE = 1.0e-4
DEFAULT_WEIGHT_TOLERANCE = 0.02

_WEIGHT_EPSILON = 1.0e-8
_MAPPING_WEIGHT_FLOOR = 1.0e-4


def _resolve_boundary_vertex_indices(obj):
    """Return the vertex indices that belong to open mesh boundaries."""
    if not is_mesh_object(obj):
        return set()

    sync_edit_mesh_data(obj)
    mesh = obj.data
    edge_face_counts = {}

    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            edge_face_counts[edge_key] = edge_face_counts.get(edge_key, 0) + 1

    boundary_indices = set()
    for edge in mesh.edges:
        edge_key = tuple(sorted(edge.vertices))
        if edge.is_loose or edge_face_counts.get(edge_key, 0) == 1:
            boundary_indices.update(edge.vertices)

    return boundary_indices


def collect_seam_vertices(obj):
    """Collect open-boundary seam vertices as sorted pairs of index and world position."""
    boundary_indices = _resolve_boundary_vertex_indices(obj)
    matrix_world = obj.matrix_world
    vertices = []
    for vertex_index in boundary_indices:
        vertex = obj.data.vertices[vertex_index]
        vertices.append((vertex.index, matrix_world @ vertex.co))
    vertices.sort(key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]))
    return vertices


def _cell_key(world_co, tolerance):
    inverse_tolerance = 1.0 / tolerance
    return (
        floor(world_co[0] * inverse_tolerance),
        floor(world_co[1] * inverse_tolerance),
        floor(world_co[2] * inverse_tolerance),
    )


def _build_spatial_hash(vertices, tolerance):
    spatial_hash = {}
    for vertex_index, world_co in vertices:
        key = _cell_key(world_co, tolerance)
        spatial_hash.setdefault(key, []).append((vertex_index, world_co))
    return spatial_hash


def _neighbor_keys(base_key):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield (base_x + offset_x, base_y + offset_y, base_z + offset_z)


def build_vertex_pairs(source_vertices, target_vertices, tolerance):
    """Build one-to-one seam vertex pairs by choosing the nearest vertex within tolerance."""
    target_spatial_hash = _build_spatial_hash(target_vertices, tolerance)
    tolerance_squared = tolerance * tolerance
    used_target_indices = set()
    matched_pairs = []
    unmatched_source = []

    for source_index, source_world_co in source_vertices:
        nearest_match = None
        nearest_distance_squared = None
        for key in _neighbor_keys(_cell_key(source_world_co, tolerance)):
            for target_index, target_world_co in target_spatial_hash.get(key, ()):
                if target_index in used_target_indices:
                    continue
                distance_squared = (source_world_co - target_world_co).length_squared
                if distance_squared > tolerance_squared:
                    continue
                if nearest_distance_squared is None or distance_squared < nearest_distance_squared:
                    nearest_match = target_index
                    nearest_distance_squared = distance_squared

        if nearest_match is None:
            unmatched_source.append(source_index)
            continue

        used_target_indices.add(nearest_match)
        matched_pairs.append((source_index, nearest_match))

    unmatched_target = [
        target_index for target_index, _world_co in target_vertices if target_index not in used_target_indices
    ]

    return matched_pairs, unmatched_source, unmatched_target


def read_vertex_weights(obj, vertex_index, group_names_by_index=None):
    """Read positive vertex-group weights from one vertex."""
    if group_names_by_index is None:
        group_names_by_index = {group.index: group.name for group in obj.vertex_groups}

    weight_map = {}
    vertex = obj.data.vertices[vertex_index]
    for assignment in vertex.groups:
        group_name = group_names_by_index.get(assignment.group)
        if group_name is None:
            continue
        if assignment.weight > _WEIGHT_EPSILON:
            weight_map[group_name] = assignment.weight
    return weight_map


@dataclass
class _WeightLoop:
    obj: object
    bm: object
    verts: list
    active: bool


def _selected_weight_loops(context):
    if context.mode != "EDIT_MESH":
        raise ValueError("Select two closed edge loops in Mesh Edit Mode.")
    loops = []
    active_obj = context.view_layer.objects.active
    for obj in edit_mesh_objects(context):
        bm = bmesh.from_edit_mesh(obj.data)
        components = selected_edge_components(bm)
        covered = set()
        for component in components:
            active_edge = active_edge_for_component(component, bm.select_history.active)
            verts, _edges = order_edge_cycle(component, active_edge)
            covered.update(verts)
            loops.append(_WeightLoop(obj, bm, verts, obj == active_obj and active_edge is not None))
        if any(vert.select and not vert.hide and vert not in covered for vert in bm.verts):
            raise ValueError("Select only the two loops; remove isolated selected vertices.")
    if len(loops) != 2:
        raise ValueError("Select exactly two closed, separate edge loops.")
    active = [loop for loop in loops if loop.active]
    # Multi-object Edit Mode has one active object. Its sole selected loop
    # identifies the reference even when loop selection left no history item.
    if not active:
        active = [loop for loop in loops if loop.obj == active_obj]
    if len(active) != 1:
        raise ValueError("Make a vertex or edge on the reference loop active.")
    target = active[0]
    source = next(loop for loop in loops if loop is not target)
    return source, target


def _loop_world_positions(loop):
    coordinates = np.asarray([tuple(vert.co) for vert in loop.verts], dtype=np.float64)
    matrix = np.asarray(loop.obj.matrix_world, dtype=np.float64)
    return coordinates @ matrix[:3, :3].T + matrix[:3, 3]


def _loop_weight_samples(source_positions, target_positions, tolerance):
    """Return reference endpoint indices and interpolation factors per source point."""
    if not np.isfinite(source_positions).all() or not np.isfinite(target_positions).all():
        raise ValueError("The selected loops contain non-finite coordinates.")
    # Center spatial queries to limit KDTree float precision loss in world space.
    origin = target_positions[0].copy()
    target = target_positions - origin
    source = source_positions - origin
    directions = np.roll(target, -1, axis=0) - target
    lengths_squared = np.einsum("ij,ij->i", directions, directions)
    midpoints = target + directions * 0.5
    max_half_length = float(np.sqrt(lengths_squared.max()) * 0.5)

    vertex_tree = KDTree(len(target))
    edge_tree = KDTree(len(target))
    for index, (position, midpoint) in enumerate(zip(target, midpoints)):
        vertex_tree.insert(tuple(position), index)
        edge_tree.insert(tuple(midpoint), index)
    vertex_tree.balance()
    edge_tree.balance()

    samples = []
    copied_count = 0
    maximum_distance = 0.0
    for point in source:
        query = tuple(point)
        _co, nearest, _distance = vertex_tree.find(query)
        nearest_distance = float(np.linalg.norm(point - target[nearest]))
        if nearest_distance <= tolerance:
            samples.append((nearest, nearest, 0.0))
            copied_count += 1
            maximum_distance = max(maximum_distance, nearest_distance)
            continue

        # Any edge closer than the nearest vertex has its midpoint within
        # nearest_distance + half its length. This bound includes long edges
        # whose endpoints are not the two nearest vertices.
        radius = nearest_distance + max_half_length
        radius += 1.0e-6 * max(radius, float(np.linalg.norm(point)), 1.0)
        candidates = sorted({item[1] for item in edge_tree.find_range(query, radius)})
        if not candidates:
            raise RuntimeError("Could not locate a reference loop segment.")
        indices = np.asarray(candidates, dtype=np.int32)
        offsets = point - target[indices]
        denominators = lengths_squared[indices]
        factors = np.divide(
            np.einsum("ij,ij->i", offsets, directions[indices]),
            denominators,
            out=np.zeros(len(indices), dtype=np.float64),
            where=denominators > 0.0,
        )
        np.clip(factors, 0.0, 1.0, out=factors)
        residuals = offsets - factors[:, None] * directions[indices]
        distances_squared = np.einsum("ij,ij->i", residuals, residuals)
        best = int(np.argmin(distances_squared))
        edge_index = int(indices[best])
        samples.append((edge_index, (edge_index + 1) % len(target), float(factors[best])))
        maximum_distance = max(maximum_distance, float(np.sqrt(distances_squared[best])))
    return samples, copied_count, maximum_distance


def _loop_named_weights(loop):
    layer = loop.bm.verts.layers.deform.active
    if layer is None:
        return [{} for _vert in loop.verts]
    names = {group.index: group.name for group in loop.obj.vertex_groups}
    return [
        {names[index]: float(weight) for index, weight in vert[layer].items()
         if index in names and weight > 0.0}
        for vert in loop.verts
    ]


def unify_selected_loop_weights(context, match_tolerance=DEFAULT_MATCH_TOLERANCE):
    """Replace non-active loop weights with samples from the active reference loop."""
    if not np.isfinite(match_tolerance) or match_tolerance < 0.0:
        raise ValueError("Coincidence tolerance must be finite and non-negative.")
    source, target = _selected_weight_loops(context)
    samples, copied_count, maximum_distance = _loop_weight_samples(
        _loop_world_positions(source), _loop_world_positions(target), match_tolerance,
    )
    reference_weights = _loop_named_weights(target)
    if not any(reference_weights):
        raise ValueError("The active reference loop has no vertex-group weights.")

    updates = []
    for first, second, factor in samples:
        if first == second:
            weights = reference_weights[first].copy()
        else:
            weights = {
                name: weight * (1.0 - factor)
                for name, weight in reference_weights[first].items()
            }
            for name, weight in reference_weights[second].items():
                weights[name] = weights.get(name, 0.0) + weight * factor
            weights = {name: weight for name, weight in weights.items() if weight > 0.0}
        updates.append(weights)

    # Adding the first deform layer can reallocate BMesh vertices. Retain
    # indices and reacquire them after group/layer creation, not RNA handles.
    source.bm.verts.index_update()
    vertex_indices = [vert.index for vert in source.verts]
    group_indices = {group.name: group.index for group in source.obj.vertex_groups}
    required_names = {name for weights in updates for name in weights}
    for name in sorted(required_names - group_indices.keys()):
        group_indices[name] = source.obj.vertex_groups.new(name=name).index
    bm = bmesh.from_edit_mesh(source.obj.data)
    layer = bm.verts.layers.deform.verify()
    bm.verts.ensure_lookup_table()
    for index, weights in zip(vertex_indices, updates):
        deform = bm.verts[index][layer]
        deform.clear()
        for name, weight in weights.items():
            deform[group_indices[name]] = weight
    bmesh.update_edit_mesh(source.obj.data, loop_triangles=False, destructive=False)
    return {
        "touched_source_vertex_count": len(updates),
        "copied_vertex_count": copied_count,
        "interpolated_vertex_count": len(updates) - copied_count,
        "empty_vertex_count": sum(not weights for weights in updates),
        "written_weight_count": sum(len(weights) for weights in updates),
        "copied_group_count": len(required_names),
        "maximum_distance": maximum_distance,
    }


def _sorted_weight_items(weight_map):
    filtered_items = [
        (group_name, weight)
        for group_name, weight in weight_map.items()
        if weight > _MAPPING_WEIGHT_FLOOR
    ]
    filtered_items.sort(key=lambda item: (-item[1], item[0]))
    return filtered_items


def _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance):
    used_target_indices = set()

    for source_group, source_weight in source_items:
        best_match_index = None
        best_match_score = -1.0
        best_match_group = None

        for target_index, (target_group, target_weight) in enumerate(target_items):
            if target_index in used_target_indices:
                continue

            difference = abs(source_weight - target_weight)
            if difference > weight_tolerance:
                continue

            similarity = 1.0 - (difference / max(weight_tolerance, _WEIGHT_EPSILON))
            match_score = min(source_weight, target_weight) * similarity
            if match_score > best_match_score:
                best_match_index = target_index
                best_match_score = match_score
                best_match_group = target_group

        if best_match_index is None or best_match_group is None:
            continue

        used_target_indices.add(best_match_index)
        mapping_key = (source_group, best_match_group)
        stats = candidate_stats.setdefault(mapping_key, {"score": 0.0, "votes": 0})
        stats["score"] += best_match_score
        stats["votes"] += 1


def build_mapping_candidates(source_obj, target_obj, matched_pairs, weight_tolerance):
    """Infer source-to-target vertex-group rename candidates from matching seam weights."""
    candidate_stats = {}
    source_group_names_by_index = {group.index: group.name for group in source_obj.vertex_groups}
    target_group_names_by_index = {group.index: group.name for group in target_obj.vertex_groups}

    for source_index, target_index in matched_pairs:
        source_items = _sorted_weight_items(
            read_vertex_weights(source_obj, source_index, source_group_names_by_index)
        )
        target_items = _sorted_weight_items(
            read_vertex_weights(target_obj, target_index, target_group_names_by_index)
        )

        if not source_items or not target_items:
            continue

        _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance)

    candidates = []
    for (source_group, target_group), stats in candidate_stats.items():
        candidates.append(
            {
                "source_group": source_group,
                "target_group": target_group,
                "score": stats["score"],
                "vote_count": stats["votes"],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["vote_count"],
            item["score"],
            item["source_group"] == item["target_group"],
        ),
        reverse=True,
    )

    used_source_groups = set()
    used_target_groups = set()
    mappings = []
    for candidate in candidates:
        if candidate["source_group"] in used_source_groups or candidate["target_group"] in used_target_groups:
            continue

        used_source_groups.add(candidate["source_group"])
        used_target_groups.add(candidate["target_group"])
        mappings.append(candidate)

    return mappings


def build_seam_rename_pairs(
    source_obj,
    target_obj,
    target_seam_vertices=None,
    match_tolerance=DEFAULT_MATCH_TOLERANCE,
    weight_tolerance=DEFAULT_WEIGHT_TOLERANCE,
):
    """Build exact source-to-target vertex-group rename pairs from paired seam vertices."""
    if not is_mesh_object(source_obj) or not is_mesh_object(target_obj):
        raise ValueError("Choose valid mesh objects.")
    if source_obj == target_obj:
        raise ValueError("Source and target objects must be different.")
    if match_tolerance <= 0.0:
        raise ValueError("Match tolerance must be greater than zero.")
    if weight_tolerance <= 0.0:
        raise ValueError("Weight tolerance must be greater than zero.")

    source_seam_vertices = collect_seam_vertices(source_obj)
    if target_seam_vertices is None:
        target_seam_vertices = collect_seam_vertices(target_obj)

    if not source_seam_vertices:
        raise ValueError(f"{source_obj.name} has no open-boundary seam vertices.")
    if not target_seam_vertices:
        raise ValueError(f"{target_obj.name} has no open-boundary seam vertices.")

    matched_pairs, unmatched_source, unmatched_target = build_vertex_pairs(
        source_seam_vertices,
        target_seam_vertices,
        match_tolerance,
    )
    if not matched_pairs:
        raise ValueError(f"{source_obj.name} has no seam vertices matching {target_obj.name}.")

    mappings = build_mapping_candidates(source_obj, target_obj, matched_pairs, weight_tolerance)
    if not mappings:
        raise ValueError(f"{source_obj.name} has no seam-weight names that match {target_obj.name}.")

    return {
        "rename_pairs": [(item["source_group"], item["target_group"]) for item in mappings],
        "mapping_count": len(mappings),
        "matched_pair_count": len(matched_pairs),
        "unmatched_source_count": len(unmatched_source),
        "unmatched_target_count": len(unmatched_target),
    }


def rename_seam_groups_to_target(
    source_obj,
    target_obj,
    target_seam_vertices=None,
    match_tolerance=DEFAULT_MATCH_TOLERANCE,
    weight_tolerance=DEFAULT_WEIGHT_TOLERANCE,
):
    """Rename source seam vertex groups to the matched names on the target object."""
    analysis = build_seam_rename_pairs(
        source_obj,
        target_obj,
        target_seam_vertices=target_seam_vertices,
        match_tolerance=match_tolerance,
        weight_tolerance=weight_tolerance,
    )
    result = apply_exact_name_pairs(source_obj, analysis["rename_pairs"])
    analysis.update(result)
    return analysis
