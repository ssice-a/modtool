"""Seam matching and weight-transfer helpers for Mod Tool."""

from __future__ import annotations

from math import floor

from .common import is_mesh_object, sync_edit_mesh_data


_WEIGHT_EPSILON = 1.0e-8
_MAPPING_WEIGHT_FLOOR = 1.0e-4


def count_enabled_group_mappings(scene):
    """Return how many stored mappings are currently enabled."""
    return sum(1 for item in scene.mdt_group_mappings if item.enabled)


def _sanitize_source_target_pair(source_obj, target_obj):
    if not is_mesh_object(source_obj):
        source_obj = None
    if not is_mesh_object(target_obj):
        target_obj = None
    if source_obj is not None and target_obj is not None and source_obj == target_obj:
        return source_obj, None
    return source_obj, target_obj


def resolve_mapping_objects(context, settings):
    """Resolve analysis meshes from the UI settings, with selection fallback."""
    source_obj = getattr(settings, "source_object", None)
    target_obj = getattr(settings, "target_object", None)
    active_obj = context.object
    selected_meshes = [obj for obj in getattr(context, "selected_objects", []) if is_mesh_object(obj)]

    if not is_mesh_object(source_obj) and is_mesh_object(active_obj):
        source_obj = active_obj

    if not is_mesh_object(target_obj):
        candidates = [obj for obj in selected_meshes if obj != source_obj]
        if len(candidates) == 1:
            target_obj = candidates[0]

    return _sanitize_source_target_pair(source_obj, target_obj)


def resolve_apply_objects(context, settings):
    """Resolve apply meshes, falling back to the analysis pair when blank."""
    apply_source_obj = getattr(settings, "apply_source_object", None)
    apply_target_obj = getattr(settings, "apply_target_object", None)

    if not is_mesh_object(apply_source_obj) or not is_mesh_object(apply_target_obj):
        fallback_source, fallback_target = resolve_mapping_objects(context, settings)
        if not is_mesh_object(apply_source_obj):
            apply_source_obj = fallback_source
        if not is_mesh_object(apply_target_obj):
            apply_target_obj = fallback_target

    return _sanitize_source_target_pair(apply_source_obj, apply_target_obj)


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


def _resolve_deform_group_names(obj):
    deform_group_names = set()
    for modifier in obj.modifiers:
        if modifier.type != "ARMATURE" or modifier.object is None:
            continue
        if getattr(modifier.object, "type", None) != "ARMATURE":
            continue
        for bone in modifier.object.data.bones:
            if bone.use_deform:
                deform_group_names.add(bone.name)
    return deform_group_names


def resolve_syncable_group_names(source_obj, target_obj, only_deform_groups):
    """Return the shared group scope used for seam analysis and apply."""
    if not only_deform_groups:
        return None

    deform_group_names = _resolve_deform_group_names(source_obj) | _resolve_deform_group_names(target_obj)
    return deform_group_names or None


def read_vertex_weights(obj, vertex_index, syncable_group_names=None):
    """Read the weight assignments for one vertex."""
    weight_map = {}
    vertex = obj.data.vertices[vertex_index]
    for assignment in vertex.groups:
        group_name = obj.vertex_groups[assignment.group].name
        if syncable_group_names is not None and group_name not in syncable_group_names:
            continue
        if assignment.weight > _WEIGHT_EPSILON:
            weight_map[group_name] = assignment.weight
    return weight_map


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


def build_mapping_candidates(source_obj, target_obj, matched_pairs, syncable_group_names, weight_tolerance):
    """Infer cross-object vertex-group mappings from nearby seam vertices with similar weights."""
    candidate_stats = {}

    for source_index, target_index in matched_pairs:
        source_weights = read_vertex_weights(source_obj, source_index, syncable_group_names)
        target_weights = read_vertex_weights(target_obj, target_index, syncable_group_names)
        source_items = _sorted_weight_items(source_weights)
        target_items = _sorted_weight_items(target_weights)

        if not source_items or not target_items:
            continue

        _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance)

    candidates = []
    for (group_a, group_b), stats in candidate_stats.items():
        candidates.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "score": stats["score"],
                "vote_count": stats["votes"],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["vote_count"],
            item["score"],
            item["group_a"] == item["group_b"],
        ),
        reverse=True,
    )

    used_group_a = set()
    used_group_b = set()
    mappings = []
    for candidate in candidates:
        if candidate["group_a"] in used_group_a or candidate["group_b"] in used_group_b:
            continue

        used_group_a.add(candidate["group_a"])
        used_group_b.add(candidate["group_b"])
        mappings.append(candidate)

    return mappings


def store_group_mappings(scene, mappings):
    """Write mapping results back to the Blender scene collection."""
    scene.mdt_group_mappings.clear()
    for mapping in mappings:
        item = scene.mdt_group_mappings.add()
        item.enabled = True
        item.group_a = mapping["group_a"]
        item.group_b = mapping["group_b"]
        item.score = mapping["score"]
        item.vote_count = mapping["vote_count"]

    if len(scene.mdt_group_mappings) == 0:
        scene.mdt_active_group_mapping_index = 0
    else:
        scene.mdt_active_group_mapping_index = min(
            scene.mdt_active_group_mapping_index,
            len(scene.mdt_group_mappings) - 1,
        )


def _get_or_create_vertex_group(obj, group_name):
    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        vertex_group = obj.vertex_groups.new(name=group_name)
    return vertex_group


def _clear_vertex_weights(obj, vertex_index, group_names):
    for group_name in group_names:
        vertex_group = obj.vertex_groups.get(group_name)
        if vertex_group is not None:
            vertex_group.remove([vertex_index])


def _write_vertex_weights(obj, vertex_index, weight_map, group_names_to_clear):
    _clear_vertex_weights(obj, vertex_index, group_names_to_clear)
    for group_name, weight in weight_map.items():
        if weight <= _WEIGHT_EPSILON:
            continue
        vertex_group = _get_or_create_vertex_group(obj, group_name)
        vertex_group.add([vertex_index], weight, "REPLACE")


def resolve_enabled_mapping_direction(scene, source_obj, invert_mapping=False):
    """Resolve stored mappings into explicit source-to-target pairs for apply."""
    resolved_mappings = []
    skipped_missing_source_groups = 0

    for item in scene.mdt_group_mappings:
        if not item.enabled or not item.group_a or not item.group_b:
            continue

        source_group = item.group_b if invert_mapping else item.group_a
        target_group = item.group_a if invert_mapping else item.group_b
        if source_obj.vertex_groups.get(source_group) is None:
            skipped_missing_source_groups += 1
            continue

        resolved_mappings.append((source_group, target_group))

    return resolved_mappings, skipped_missing_source_groups


def apply_weights_to_vertex(
    source_obj,
    source_index,
    target_obj,
    target_index,
    mappings,
    target_groups_to_clear,
    syncable_group_names=None,
):
    """Copy mapped source seam weights to the paired vertex on the target seam."""
    source_weights = read_vertex_weights(source_obj, source_index, syncable_group_names)
    payload = {}

    for source_group, target_group in mappings:
        weight = source_weights.get(source_group, 0.0)
        if weight > _WEIGHT_EPSILON:
            payload[target_group] = weight

    _write_vertex_weights(target_obj, target_index, payload, target_groups_to_clear)
