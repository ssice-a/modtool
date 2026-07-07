"""Seam-based vertex-group rename helpers for Mod Tool."""

from __future__ import annotations

from math import floor

from .common import is_mesh_object, sync_edit_mesh_data
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


def _ensure_vertex_groups(obj, group_names):
    groups_by_name = {group.name: group for group in obj.vertex_groups}
    for group_name in group_names:
        if group_name not in groups_by_name:
            groups_by_name[group_name] = obj.vertex_groups.new(name=group_name)
    return groups_by_name


def _build_seam_weight_update_plan(source_obj, target_obj, matched_pairs):
    source_group_names_by_index = {group.index: group.name for group in source_obj.vertex_groups}
    target_group_names_by_index = {group.index: group.name for group in target_obj.vertex_groups}
    removals_by_group_name = {}
    additions_by_group_name = {}
    copied_target_group_names = set()

    for source_index, target_index in matched_pairs:
        source_vertex = source_obj.data.vertices[source_index]
        for assignment in source_vertex.groups:
            if assignment.weight <= _WEIGHT_EPSILON:
                continue
            group_name = source_group_names_by_index.get(assignment.group)
            if group_name is None:
                continue
            removals_by_group_name.setdefault(group_name, []).append(source_index)

        target_weights = read_vertex_weights(target_obj, target_index, target_group_names_by_index)
        for group_name, weight in target_weights.items():
            if weight <= _WEIGHT_EPSILON:
                continue
            additions_by_group_name.setdefault(group_name, []).append((source_index, min(1.0, float(weight))))
            copied_target_group_names.add(group_name)

    return removals_by_group_name, additions_by_group_name, copied_target_group_names


def _apply_seam_weight_update_plan(source_obj, removals_by_group_name, additions_by_group_name):
    existing_groups_by_name = {group.name: group for group in source_obj.vertex_groups}
    for group_name, vertex_indices in removals_by_group_name.items():
        group = existing_groups_by_name.get(group_name)
        if group is None or not vertex_indices:
            continue
        group.remove(tuple(vertex_indices))

    groups_by_name = _ensure_vertex_groups(source_obj, additions_by_group_name)
    written_weight_count = 0
    for group_name, additions in additions_by_group_name.items():
        group = groups_by_name[group_name]
        for vertex_index, weight in additions:
            group.add((vertex_index,), weight, "REPLACE")
            written_weight_count += 1
    return written_weight_count


def unify_seam_vertex_weights(
    source_obj,
    target_obj,
    source_seam_vertices=None,
    target_seam_vertices=None,
    match_tolerance=DEFAULT_MATCH_TOLERANCE,
):
    """Copy matched target seam vertex weights onto source seam vertices."""
    if not is_mesh_object(source_obj) or not is_mesh_object(target_obj):
        raise ValueError("Choose valid mesh objects.")
    if source_obj == target_obj:
        raise ValueError("Source and target objects must be different.")
    if match_tolerance <= 0.0:
        raise ValueError("Match tolerance must be greater than zero.")

    if source_seam_vertices is None:
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

    removals_by_group_name, additions_by_group_name, copied_target_group_names = _build_seam_weight_update_plan(
        source_obj,
        target_obj,
        matched_pairs,
    )
    written_weight_count = _apply_seam_weight_update_plan(
        source_obj,
        removals_by_group_name,
        additions_by_group_name,
    )

    return {
        "matched_pair_count": len(matched_pairs),
        "unmatched_source_count": len(unmatched_source),
        "unmatched_target_count": len(unmatched_target),
        "touched_source_vertex_count": len(matched_pairs),
        "written_weight_count": written_weight_count,
        "copied_group_count": len(copied_target_group_names),
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
