"""Shape-key transfer helpers for Mod Tool."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import bpy
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .common import is_mesh_object, sync_edit_mesh_data


_TRIANGLE_RELATIVE_EPSILON = 1.0e-12
_VECTOR_EPSILON = 1.0e-8
_VERTEX_CANDIDATE_LIMIT = 16
_TEMP_SURFACE_DEFORM_NAME = "MDT_WrapBake"


@dataclass
class SurfaceSample:
    """One cached shared-motion sample on the source surface."""

    source_indices: tuple[int, int, int]
    weights: tuple[float, float, float]
    source_basis_point_local: object
    target_vertex_indices: tuple[int, ...]


@dataclass
class VertexSample:
    """One cached shared-motion sample driven by a source vertex."""

    source_vertex_index: int
    target_vertex_indices: tuple[int, ...]


def has_transferable_shape_keys(obj):
    """Return True when the mesh has at least one non-basis shape key."""
    if not is_mesh_object(obj):
        return False

    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None:
        return False

    return len(shape_keys.key_blocks) > 1


def resolve_transfer_objects(context, settings):
    """Resolve source and target meshes for shape-key transfer with light selection fallback."""
    source_obj = getattr(settings, "source_object", None)
    target_obj = getattr(settings, "target_object", None)
    active_obj = context.object
    selected_meshes = [obj for obj in getattr(context, "selected_objects", []) if is_mesh_object(obj)]

    if not has_transferable_shape_keys(source_obj) and has_transferable_shape_keys(active_obj):
        source_obj = active_obj

    if not is_mesh_object(target_obj):
        candidates = [obj for obj in selected_meshes if obj != source_obj]
        if len(candidates) == 1:
            target_obj = candidates[0]

    if not is_mesh_object(source_obj):
        source_obj = None
    if not is_mesh_object(target_obj):
        target_obj = None
    if source_obj is not None and target_obj is not None and source_obj == target_obj:
        return source_obj, None
    return source_obj, target_obj


def _get_basis_coordinates(obj):
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is not None and len(shape_keys.key_blocks) > 0:
        return [point.co.copy() for point in shape_keys.key_blocks[0].data]
    return [vertex.co.copy() for vertex in obj.data.vertices]


def _build_loop_triangles(obj):
    sync_edit_mesh_data(obj)
    mesh = obj.data
    mesh.calc_loop_triangles()
    return [tuple(loop_triangle.vertices) for loop_triangle in mesh.loop_triangles]


def _build_polygons(obj):
    sync_edit_mesh_data(obj)
    return [tuple(polygon.vertices) for polygon in obj.data.polygons]


def _build_vertex_polygon_indices(polygons, vertex_count):
    polygon_indices_by_vertex = [[] for _ in range(vertex_count)]
    for polygon_index, polygon in enumerate(polygons):
        for vertex_index in polygon:
            polygon_indices_by_vertex[vertex_index].append(polygon_index)
    return polygon_indices_by_vertex


def _build_kdtree(world_vertices):
    kd_tree = KDTree(len(world_vertices))
    for vertex_index, world_coordinate in enumerate(world_vertices):
        kd_tree.insert(world_coordinate, vertex_index)
    kd_tree.balance()
    return kd_tree


def _polygon_centroid(world_coordinates):
    centroid = world_coordinates[0].copy()
    for coordinate in world_coordinates[1:]:
        centroid += coordinate
    return centroid / len(world_coordinates)


def _polygon_normal(world_coordinates):
    normal = world_coordinates[0].copy() * 0.0
    polygon_size = len(world_coordinates)
    for index, current_coordinate in enumerate(world_coordinates):
        next_coordinate = world_coordinates[(index + 1) % polygon_size]
        normal.x += (current_coordinate.y - next_coordinate.y) * (current_coordinate.z + next_coordinate.z)
        normal.y += (current_coordinate.z - next_coordinate.z) * (current_coordinate.x + next_coordinate.x)
        normal.z += (current_coordinate.x - next_coordinate.x) * (current_coordinate.y + next_coordinate.y)

    if normal.length <= _VECTOR_EPSILON:
        return normal
    normal.normalize()
    return normal


def _build_vertex_feature_data(obj, world_vertices):
    polygons = _build_polygons(obj)
    vertex_face_offsets = [[] for _ in world_vertices]
    vertex_face_normals = [[] for _ in world_vertices]
    vertex_face_sizes = [[] for _ in world_vertices]
    vertex_neighbor_indices = [set() for _ in world_vertices]

    for polygon in polygons:
        polygon_world_coordinates = [world_vertices[vertex_index] for vertex_index in polygon]
        polygon_centroid = _polygon_centroid(polygon_world_coordinates)
        polygon_normal = _polygon_normal(polygon_world_coordinates)
        polygon_size = len(polygon)

        for loop_index, vertex_index in enumerate(polygon):
            vertex_world_coordinate = world_vertices[vertex_index]
            vertex_face_offsets[vertex_index].append(polygon_centroid - vertex_world_coordinate)
            vertex_face_normals[vertex_index].append(polygon_normal.copy())
            vertex_face_sizes[vertex_index].append(polygon_size)

            previous_vertex_index = polygon[loop_index - 1]
            next_vertex_index = polygon[(loop_index + 1) % polygon_size]
            vertex_neighbor_indices[vertex_index].add(previous_vertex_index)
            vertex_neighbor_indices[vertex_index].add(next_vertex_index)

    features = []
    for vertex_index, vertex_world_coordinate in enumerate(world_vertices):
        neighbor_offsets = [
            world_vertices[neighbor_index] - vertex_world_coordinate
            for neighbor_index in vertex_neighbor_indices[vertex_index]
        ]
        features.append(
            {
                "face_count": len(vertex_face_offsets[vertex_index]),
                "neighbor_count": len(neighbor_offsets),
                "face_offsets": vertex_face_offsets[vertex_index],
                "face_normals": vertex_face_normals[vertex_index],
                "neighbor_offsets": neighbor_offsets,
                "face_sizes": sorted(vertex_face_sizes[vertex_index]),
            }
        )

    return features


def _blend_triangle_coordinates(coordinates, indices, weights):
    def _resolve_coordinate(index):
        coordinate = coordinates[index]
        return coordinate.co if hasattr(coordinate, "co") else coordinate

    index_a, index_b, index_c = indices
    weight_a, weight_b, weight_c = weights
    return (
        _resolve_coordinate(index_a) * weight_a
        + _resolve_coordinate(index_b) * weight_b
        + _resolve_coordinate(index_c) * weight_c
    )


def _vector_distance(vector_a, vector_b, normalize=False):
    if not normalize:
        return (vector_a - vector_b).length

    length_a = vector_a.length
    length_b = vector_b.length
    if length_a <= _VECTOR_EPSILON and length_b <= _VECTOR_EPSILON:
        return 0.0
    if length_a <= _VECTOR_EPSILON or length_b <= _VECTOR_EPSILON:
        return 2.0
    return (vector_a.normalized() - vector_b.normalized()).length


def _greedy_vector_match_score(target_vectors, source_vectors, normalize=False):
    if not target_vectors and not source_vectors:
        return 0.0

    remaining_source_indices = list(range(len(source_vectors)))
    total_score = 0.0

    for target_vector in sorted(target_vectors, key=lambda item: item.length, reverse=True):
        if not remaining_source_indices:
            total_score += max(1.0, target_vector.length) * 4.0
            continue

        best_source_index = min(
            remaining_source_indices,
            key=lambda source_index: _vector_distance(target_vector, source_vectors[source_index], normalize),
        )
        total_score += _vector_distance(target_vector, source_vectors[best_source_index], normalize)
        remaining_source_indices.remove(best_source_index)

    for source_index in remaining_source_indices:
        total_score += max(1.0, source_vectors[source_index].length) * 2.0

    return total_score


def _face_size_score(target_feature, source_feature):
    target_face_sizes = target_feature["face_sizes"]
    source_face_sizes = source_feature["face_sizes"]
    pair_count = min(len(target_face_sizes), len(source_face_sizes))
    score = sum(
        abs(target_face_sizes[index] - source_face_sizes[index])
        for index in range(pair_count)
    )
    score += abs(len(target_face_sizes) - len(source_face_sizes)) * 8.0
    return score


def _vertex_feature_score(target_feature, source_feature, position_distance):
    score = position_distance * 5000.0
    score += abs(target_feature["face_count"] - source_feature["face_count"]) * 200.0
    score += abs(target_feature["neighbor_count"] - source_feature["neighbor_count"]) * 120.0
    score += _face_size_score(target_feature, source_feature) * 40.0
    score += _greedy_vector_match_score(target_feature["face_offsets"], source_feature["face_offsets"]) * 10.0
    score += _greedy_vector_match_score(target_feature["neighbor_offsets"], source_feature["neighbor_offsets"]) * 15.0
    score += _greedy_vector_match_score(
        target_feature["face_normals"],
        source_feature["face_normals"],
        normalize=True,
    ) * 30.0
    return score


def _vertex_face_hint_score(
    source_vertex_index,
    source_vertex_to_group,
    face_group_hints,
    face_vertex_hints,
    considered_face_count,
):
    score = 0.0
    source_group_index = source_vertex_to_group[source_vertex_index]
    group_hint = face_group_hints.get(source_group_index)
    if group_hint is not None:
        support_ratio = group_hint["votes"] / max(1, considered_face_count)
        average_distance = group_hint["distance_sum"] / max(1, group_hint["votes"])
        score += support_ratio * 3.0
        score += 3.0 / (1.0 + (average_distance * 2000.0))

    vertex_hint = face_vertex_hints.get(source_vertex_index)
    if vertex_hint is not None:
        average_distance = vertex_hint["distance_sum"] / max(1, vertex_hint["votes"])
        score += min(3, vertex_hint["votes"]) * 1.5
        score += 1.5 / (1.0 + (average_distance * 2000.0))

    return score


def _cell_key(world_coordinate, tolerance):
    inverse_tolerance = 1.0 / tolerance
    return (
        floor(world_coordinate[0] * inverse_tolerance),
        floor(world_coordinate[1] * inverse_tolerance),
        floor(world_coordinate[2] * inverse_tolerance),
    )


def _neighbor_keys(base_key):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield (base_x + offset_x, base_y + offset_y, base_z + offset_z)


def _triangle_denominator(tri_a, tri_b, tri_c):
    edge_ab = tri_b - tri_a
    edge_ac = tri_c - tri_a
    dot_ab_ab = edge_ab.dot(edge_ab)
    dot_ab_ac = edge_ab.dot(edge_ac)
    dot_ac_ac = edge_ac.dot(edge_ac)
    denominator = (dot_ab_ab * dot_ac_ac) - (dot_ab_ac * dot_ab_ac)
    scale = max(dot_ab_ab * dot_ac_ac, 1.0)
    return denominator, scale


def _is_degenerate_triangle(tri_a, tri_b, tri_c):
    denominator, scale = _triangle_denominator(tri_a, tri_b, tri_c)
    return abs(denominator) <= (scale * _TRIANGLE_RELATIVE_EPSILON)


def _barycentric_weights(point, tri_a, tri_b, tri_c):
    edge_ab = tri_b - tri_a
    edge_ac = tri_c - tri_a
    offset = point - tri_a

    dot_ab_ab = edge_ab.dot(edge_ab)
    dot_ab_ac = edge_ab.dot(edge_ac)
    dot_ac_ac = edge_ac.dot(edge_ac)
    dot_offset_ab = offset.dot(edge_ab)
    dot_offset_ac = offset.dot(edge_ac)

    denominator = (dot_ab_ab * dot_ac_ac) - (dot_ab_ac * dot_ab_ac)
    scale = max(dot_ab_ab * dot_ac_ac, 1.0)
    if abs(denominator) <= (scale * _TRIANGLE_RELATIVE_EPSILON):
        return None

    weight_b = ((dot_ac_ac * dot_offset_ab) - (dot_ab_ac * dot_offset_ac)) / denominator
    weight_c = ((dot_ab_ab * dot_offset_ac) - (dot_ab_ac * dot_offset_ab)) / denominator
    weight_a = 1.0 - weight_b - weight_c
    return (weight_a, weight_b, weight_c)


def _filter_valid_source_triangles(source_world_vertices, source_triangles):
    valid_triangles = []
    ignored_degenerate_triangles = 0

    for triangle in source_triangles:
        tri_a, tri_b, tri_c = [source_world_vertices[index] for index in triangle]
        if _is_degenerate_triangle(tri_a, tri_b, tri_c):
            ignored_degenerate_triangles += 1
            continue
        valid_triangles.append(triangle)

    return valid_triangles, ignored_degenerate_triangles


def _group_vertices_by_position(world_vertices, tolerance):
    groups = []
    grouped_vertex_count = 0
    vertex_to_group = [-1] * len(world_vertices)

    if tolerance <= 0.0:
        groups_by_coordinate = {}
        ordered_keys = []
        for vertex_index, world_coordinate in enumerate(world_vertices):
            key = (
                float(world_coordinate[0]),
                float(world_coordinate[1]),
                float(world_coordinate[2]),
            )
            if key not in groups_by_coordinate:
                groups_by_coordinate[key] = {"vertex_indices": [], "group_index": len(ordered_keys)}
                ordered_keys.append(key)
            groups_by_coordinate[key]["vertex_indices"].append(vertex_index)
            vertex_to_group[vertex_index] = groups_by_coordinate[key]["group_index"]

        for key in ordered_keys:
            vertex_indices = groups_by_coordinate[key]["vertex_indices"]
            grouped_vertex_count += max(0, len(vertex_indices) - 1)
            groups.append(
                {
                    "vertex_indices": tuple(vertex_indices),
                    "representative_world_coordinate": world_vertices[vertex_indices[0]].copy(),
                }
            )
        return groups, grouped_vertex_count, vertex_to_group

    spatial_hash = {}
    for vertex_index, world_coordinate in enumerate(world_vertices):
        matched_group = None
        matched_group_index = None
        for neighbor_key in _neighbor_keys(_cell_key(world_coordinate, tolerance)):
            for group_index in spatial_hash.get(neighbor_key, ()):
                group = groups[group_index]
                if (
                    world_coordinate - group["representative_world_coordinate"]
                ).length <= tolerance:
                    matched_group = group
                    matched_group_index = group_index
                    break
            if matched_group is not None:
                break

        if matched_group is None:
            group_index = len(groups)
            groups.append(
                {
                    "vertex_indices": [vertex_index],
                    "representative_world_coordinate": world_coordinate.copy(),
                }
            )
            vertex_to_group[vertex_index] = group_index
            spatial_hash.setdefault(_cell_key(world_coordinate, tolerance), []).append(group_index)
            continue

        matched_group["vertex_indices"].append(vertex_index)
        vertex_to_group[vertex_index] = matched_group_index
        grouped_vertex_count += 1

    for group in groups:
        group["vertex_indices"] = tuple(group["vertex_indices"])

    return groups, grouped_vertex_count, vertex_to_group


def _align_polygon_vertex_order(source_polygon, target_polygon, source_world_vertices, target_world_vertices):
    source_polygon = tuple(source_polygon)
    target_polygon = tuple(target_polygon)
    polygon_size = len(source_polygon)
    if polygon_size != len(target_polygon):
        raise ValueError("Source and target polygons do not have the same corner count.")

    best_source_order = None
    best_score = None
    best_max_distance = 0.0

    for reverse in (False, True):
        candidate_source = tuple(reversed(source_polygon)) if reverse else source_polygon
        for rotation in range(polygon_size):
            rotated_source = tuple(
                candidate_source[(corner_index + rotation) % polygon_size]
                for corner_index in range(polygon_size)
            )
            score = 0.0
            max_distance = 0.0
            for source_index, target_index in zip(rotated_source, target_polygon):
                corner_distance = (source_world_vertices[source_index] - target_world_vertices[target_index]).length
                score += corner_distance * corner_distance
                max_distance = max(max_distance, corner_distance)

            if best_score is None or score < best_score:
                best_score = score
                best_source_order = rotated_source
                best_max_distance = max_distance

    return best_source_order, best_max_distance, best_score


def _collect_group_face_hints(
    target_group,
    source_polygons,
    target_polygons,
    source_world_vertices,
    target_world_vertices,
    source_vertex_to_group,
    target_polygon_indices_by_vertex,
):
    face_group_hints = {}
    face_vertex_hints = {}
    considered_face_count = 0
    max_corner_distance = 0.0

    shared_polygon_count = min(len(source_polygons), len(target_polygons))
    for target_vertex_index in target_group["vertex_indices"]:
        for polygon_index in target_polygon_indices_by_vertex[target_vertex_index]:
            if polygon_index >= shared_polygon_count:
                continue

            source_polygon = source_polygons[polygon_index]
            target_polygon = target_polygons[polygon_index]
            if len(source_polygon) != len(target_polygon):
                continue

            aligned_source_polygon, polygon_max_distance, _alignment_score = _align_polygon_vertex_order(
                source_polygon,
                target_polygon,
                source_world_vertices,
                target_world_vertices,
            )
            considered_face_count += 1
            max_corner_distance = max(max_corner_distance, polygon_max_distance)

            for source_vertex_index, polygon_target_vertex_index in zip(aligned_source_polygon, target_polygon):
                if polygon_target_vertex_index != target_vertex_index:
                    continue

                corner_distance = (
                    source_world_vertices[source_vertex_index] - target_world_vertices[target_vertex_index]
                ).length
                source_group_index = source_vertex_to_group[source_vertex_index]

                group_hint = face_group_hints.setdefault(source_group_index, {"votes": 0, "distance_sum": 0.0})
                group_hint["votes"] += 1
                group_hint["distance_sum"] += corner_distance

                vertex_hint = face_vertex_hints.setdefault(source_vertex_index, {"votes": 0, "distance_sum": 0.0})
                vertex_hint["votes"] += 1
                vertex_hint["distance_sum"] += corner_distance
                break

    return face_group_hints, face_vertex_hints, considered_face_count, max_corner_distance


def build_vertex_samples(source_obj, target_obj, vertex_tolerance, shared_vertex_tolerance=0.0):
    """Map grouped target vertices onto source vertices using spatial search plus weak polygon hints."""
    if vertex_tolerance < 0.0:
        raise ValueError("Vertex Tolerance must be zero or greater.")
    if shared_vertex_tolerance < 0.0:
        raise ValueError("Shared Vertex Tolerance must be zero or greater.")

    source_basis = _get_basis_coordinates(source_obj)
    target_basis = _get_basis_coordinates(target_obj)
    if not source_basis:
        raise ValueError("The source mesh has no vertices.")
    if not target_basis:
        raise ValueError("The target mesh has no vertices.")

    source_world_vertices = [source_obj.matrix_world @ coordinate for coordinate in source_basis]
    target_world_vertices = [target_obj.matrix_world @ coordinate for coordinate in target_basis]
    source_features = _build_vertex_feature_data(source_obj, source_world_vertices)
    target_features = _build_vertex_feature_data(target_obj, target_world_vertices)
    source_polygons = _build_polygons(source_obj)
    target_polygons = _build_polygons(target_obj)
    _source_groups, _source_grouped_count, source_vertex_to_group = _group_vertices_by_position(
        source_world_vertices,
        shared_vertex_tolerance,
    )
    target_groups, grouped_vertex_count, _target_vertex_to_group = _group_vertices_by_position(
        target_world_vertices,
        shared_vertex_tolerance,
    )
    target_polygon_indices_by_vertex = _build_vertex_polygon_indices(target_polygons, len(target_world_vertices))
    source_kd_tree = _build_kdtree(source_world_vertices)

    samples = []
    unmatched_count = 0
    ambiguous_group_count = 0
    face_guided_group_count = 0
    farthest_distance = 0.0
    max_face_corner_distance = 0.0

    for target_group in target_groups:
        representative_target_vertex_index = target_group["vertex_indices"][0]
        representative_target_coordinate = target_world_vertices[representative_target_vertex_index]
        face_group_hints, face_vertex_hints, considered_face_count, face_corner_distance = _collect_group_face_hints(
            target_group,
            source_polygons,
            target_polygons,
            source_world_vertices,
            target_world_vertices,
            source_vertex_to_group,
            target_polygon_indices_by_vertex,
        )
        max_face_corner_distance = max(max_face_corner_distance, face_corner_distance)
        nearest_candidates = source_kd_tree.find_n(
            representative_target_coordinate,
            min(_VERTEX_CANDIDATE_LIMIT, len(source_world_vertices)),
        )
        if not nearest_candidates:
            unmatched_count += len(target_group["vertex_indices"])
            continue

        nearby_candidates = [
            (source_vertex_index, float(distance))
            for _world_coordinate, source_vertex_index, distance in nearest_candidates
            if distance <= vertex_tolerance
        ]
        if not nearby_candidates:
            farthest_distance = max(farthest_distance, float(nearest_candidates[0][2]))
            unmatched_count += len(target_group["vertex_indices"])
            continue

        target_feature = target_features[representative_target_vertex_index]
        base_scored_candidates = []
        for source_vertex_index, distance in nearby_candidates:
            source_feature = source_features[source_vertex_index]
            base_score = _vertex_feature_score(target_feature, source_feature, distance)
            face_hint_score = _vertex_face_hint_score(
                source_vertex_index,
                source_vertex_to_group,
                face_group_hints,
                face_vertex_hints,
                considered_face_count,
            )
            base_scored_candidates.append(
                (
                    base_score,
                    source_vertex_index,
                    distance,
                    face_hint_score,
                )
            )

        base_scored_candidates.sort(key=lambda item: (item[0], item[2], item[1]))
        best_base_score, best_base_source_vertex_index, _best_base_distance, _best_base_face_hint_score = (
            base_scored_candidates[0]
        )
        shortlist_margin = max(2.0, best_base_score * 0.1)
        shortlisted_candidates = [
            candidate
            for candidate in base_scored_candidates
            if candidate[0] <= (best_base_score + shortlist_margin)
        ][:4]

        scored_candidates = [
            (
                base_score - face_hint_score,
                base_score,
                distance,
                source_vertex_index,
                face_hint_score,
            )
            for base_score, source_vertex_index, distance, face_hint_score in shortlisted_candidates
        ]
        scored_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_score, _best_base_score, best_distance, best_source_vertex_index, best_face_hint_score = scored_candidates[0]
        if best_source_vertex_index != best_base_source_vertex_index and best_face_hint_score > 0.0:
            face_guided_group_count += 1
        if len(scored_candidates) > 1:
            second_score = scored_candidates[1][0]
            if abs(second_score - best_score) <= 1.0e-6 or second_score <= best_score * 1.02:
                ambiguous_group_count += 1

        samples.append(
            VertexSample(
                source_vertex_index=best_source_vertex_index,
                target_vertex_indices=target_group["vertex_indices"],
            )
        )
        farthest_distance = max(farthest_distance, best_distance)

    return {
        "samples": samples,
        "target_vertex_count": len(target_basis),
        "target_group_count": len(target_groups),
        "unmatched_count": unmatched_count,
        "matched_vertex_count": len(target_basis) - unmatched_count,
        "grouped_vertex_count": grouped_vertex_count,
        "ambiguous_group_count": ambiguous_group_count,
        "face_guided_group_count": face_guided_group_count,
        "max_face_corner_distance": max_face_corner_distance,
        "farthest_distance": farthest_distance,
    }


def build_surface_samples(source_obj, target_obj, max_distance, shared_vertex_tolerance=0.0):
    """Map grouped target-basis vertices onto the source surface and cache interpolation data."""
    if max_distance < 0.0:
        raise ValueError("Surface Tolerance must be zero or greater.")
    if shared_vertex_tolerance < 0.0:
        raise ValueError("Shared Vertex Tolerance must be zero or greater.")

    source_basis = _get_basis_coordinates(source_obj)
    target_basis = _get_basis_coordinates(target_obj)
    source_triangles = _build_loop_triangles(source_obj)

    if not source_basis:
        raise ValueError("The source mesh has no vertices.")
    if not target_basis:
        raise ValueError("The target mesh has no vertices.")
    if not source_triangles:
        raise ValueError("The source mesh has no faces to sample.")

    source_world_matrix = source_obj.matrix_world
    target_world_matrix = target_obj.matrix_world
    source_world_vertices = [source_world_matrix @ coordinate for coordinate in source_basis]
    valid_source_triangles, ignored_degenerate_triangles = _filter_valid_source_triangles(
        source_world_vertices,
        source_triangles,
    )
    if not valid_source_triangles:
        raise ValueError("The source mesh has no usable non-degenerate faces to sample.")

    bvh_tree = BVHTree.FromPolygons(source_world_vertices, valid_source_triangles, all_triangles=True)
    if bvh_tree is None:
        raise ValueError("Could not build a source surface search tree.")

    target_world_vertices = [target_world_matrix @ coordinate for coordinate in target_basis]
    target_groups, grouped_vertex_count, _target_vertex_to_group = _group_vertices_by_position(
        target_world_vertices,
        shared_vertex_tolerance,
    )

    samples = []
    unmatched_count = 0
    farthest_distance = 0.0

    for target_group in target_groups:
        target_vertex_indices = target_group["vertex_indices"]
        target_world_coordinate = target_group["representative_world_coordinate"]
        nearest = bvh_tree.find_nearest(target_world_coordinate)
        if nearest is None:
            unmatched_count += len(target_vertex_indices)
            continue

        hit_coordinate, _normal, triangle_index, distance = nearest
        if triangle_index is None or distance is None:
            unmatched_count += len(target_vertex_indices)
            continue

        farthest_distance = max(farthest_distance, float(distance))
        if distance > max_distance:
            unmatched_count += len(target_vertex_indices)
            continue

        source_indices = valid_source_triangles[triangle_index]
        tri_a, tri_b, tri_c = [source_world_vertices[index] for index in source_indices]
        weights = _barycentric_weights(hit_coordinate, tri_a, tri_b, tri_c)
        if weights is None:
            unmatched_count += len(target_vertex_indices)
            continue

        samples.append(
            SurfaceSample(
                source_indices=source_indices,
                weights=weights,
                source_basis_point_local=_blend_triangle_coordinates(source_basis, source_indices, weights),
                target_vertex_indices=target_vertex_indices,
            )
        )

    return {
        "samples": samples,
        "target_vertex_count": len(target_basis),
        "target_group_count": len(target_groups),
        "unmatched_count": unmatched_count,
        "matched_vertex_count": len(target_basis) - unmatched_count,
        "grouped_vertex_count": grouped_vertex_count,
        "ignored_degenerate_triangle_count": ignored_degenerate_triangles,
        "farthest_distance": farthest_distance,
    }


def ensure_basis_shape_key(obj):
    """Ensure the target mesh has a basis shape key and return it."""
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is not None and len(shape_keys.key_blocks) > 0:
        return shape_keys.key_blocks[0]
    return obj.shape_key_add(name="Basis", from_mix=False)


def remove_shape_key_by_name(obj, key_name):
    """Remove a target shape key by name when it already exists."""
    shape_keys = getattr(obj.data, "shape_keys", None)
    if shape_keys is None:
        return False

    key_block = shape_keys.key_blocks.get(key_name)
    if key_block is None or key_block == shape_keys.key_blocks[0]:
        return False

    obj.shape_key_remove(key_block)
    return True


def _copy_shape_key_metadata(source_key, target_key, value_override=None, mute_override=None):
    target_key.slider_min = source_key.slider_min
    target_key.slider_max = source_key.slider_max
    target_key.value = source_key.value if value_override is None else value_override
    target_key.mute = source_key.mute if mute_override is None else mute_override
    target_key.vertex_group = source_key.vertex_group
    target_key.interpolation = source_key.interpolation


def _apply_relative_keys(source_shape_keys, target_basis_key, resolved_target_keys, relative_names):
    basis_name = source_shape_keys.key_blocks[0].name
    target_shape_keys = getattr(target_basis_key.id_data, "key_blocks", None)

    for key_name, relative_name in relative_names.items():
        target_key = resolved_target_keys.get(key_name)
        if target_key is None:
            continue

        if not relative_name or relative_name == basis_name:
            target_key.relative_key = target_basis_key
            continue

        target_key.relative_key = (
            resolved_target_keys.get(relative_name)
            or (target_shape_keys.get(relative_name) if target_shape_keys is not None else None)
            or target_basis_key
        )


def transfer_shape_keys(source_obj, target_obj, samples, replace_existing=True):
    """Transfer every non-basis shape key from source to target through cached surface samples."""
    source_shape_keys = getattr(source_obj.data, "shape_keys", None)
    if source_shape_keys is None or len(source_shape_keys.key_blocks) <= 1:
        raise ValueError("The source object has no transferable shape keys.")
    if len(samples) == 0:
        raise ValueError("There are no cached target samples to transfer.")

    target_basis_key = ensure_basis_shape_key(target_obj)
    target_basis = [point.co.copy() for point in target_basis_key.data]
    source_linear = source_obj.matrix_world.to_3x3()
    target_linear_inverse = target_obj.matrix_world.inverted().to_3x3()

    created_count = 0
    replaced_count = 0
    skipped_count = 0
    relative_names = {}
    created_keys = {}

    for source_key in source_shape_keys.key_blocks[1:]:
        if target_obj.data.shape_keys is not None and target_obj.data.shape_keys.key_blocks.get(source_key.name):
            if replace_existing:
                if remove_shape_key_by_name(target_obj, source_key.name):
                    replaced_count += 1
            else:
                skipped_count += 1
                continue

        target_key = target_obj.shape_key_add(name=source_key.name, from_mix=False)
        _copy_shape_key_metadata(source_key, target_key)

        for sample in samples:
            source_key_point_local = _blend_triangle_coordinates(
                source_key.data,
                sample.source_indices,
                sample.weights,
            )
            source_delta_world = source_linear @ (source_key_point_local - sample.source_basis_point_local)
            target_delta_local = target_linear_inverse @ source_delta_world
            for target_vertex_index in sample.target_vertex_indices:
                target_key.data[target_vertex_index].co = target_basis[target_vertex_index] + target_delta_local

        relative_key = source_key.relative_key
        relative_names[source_key.name] = relative_key.name if relative_key is not None else ""
        created_keys[source_key.name] = target_key
        created_count += 1

    _apply_relative_keys(source_shape_keys, target_basis_key, created_keys, relative_names)
    return {
        "created_count": created_count,
        "replaced_count": replaced_count,
        "skipped_count": skipped_count,
    }


def transfer_shape_keys_by_vertex(source_obj, target_obj, samples, replace_existing=True):
    """Transfer shape keys by directly matching each target group to one source vertex."""
    source_shape_keys = getattr(source_obj.data, "shape_keys", None)
    if source_shape_keys is None or len(source_shape_keys.key_blocks) <= 1:
        raise ValueError("The source object has no transferable shape keys.")
    if len(samples) == 0:
        raise ValueError("There are no cached vertex matches to transfer.")

    target_basis_key = ensure_basis_shape_key(target_obj)
    source_basis = _get_basis_coordinates(source_obj)
    target_basis = [point.co.copy() for point in target_basis_key.data]
    source_linear = source_obj.matrix_world.to_3x3()
    target_linear_inverse = target_obj.matrix_world.inverted().to_3x3()

    created_count = 0
    replaced_count = 0
    skipped_count = 0
    relative_names = {}
    created_keys = {}

    for source_key in source_shape_keys.key_blocks[1:]:
        if target_obj.data.shape_keys is not None and target_obj.data.shape_keys.key_blocks.get(source_key.name):
            if replace_existing:
                if remove_shape_key_by_name(target_obj, source_key.name):
                    replaced_count += 1
            else:
                skipped_count += 1
                continue

        target_key = target_obj.shape_key_add(name=source_key.name, from_mix=False)
        _copy_shape_key_metadata(source_key, target_key)

        for sample in samples:
            source_vertex_index = sample.source_vertex_index
            source_delta_local = source_key.data[source_vertex_index].co - source_basis[source_vertex_index]
            source_delta_world = source_linear @ source_delta_local
            target_delta_local = target_linear_inverse @ source_delta_world
            for target_vertex_index in sample.target_vertex_indices:
                target_key.data[target_vertex_index].co = target_basis[target_vertex_index] + target_delta_local

        relative_key = source_key.relative_key
        relative_names[source_key.name] = relative_key.name if relative_key is not None else ""
        created_keys[source_key.name] = target_key
        created_count += 1

    _apply_relative_keys(source_shape_keys, target_basis_key, created_keys, relative_names)
    return {
        "created_count": created_count,
        "replaced_count": replaced_count,
        "skipped_count": skipped_count,
    }


def _capture_source_runtime_state(source_obj):
    shape_keys = getattr(source_obj.data, "shape_keys", None)
    if shape_keys is None:
        return None

    return {
        "active_shape_key_index": getattr(source_obj, "active_shape_key_index", 0),
        "show_only_shape_key": bool(getattr(source_obj, "show_only_shape_key", False)),
        "key_values": {
            key_block.name: {"value": key_block.value, "mute": key_block.mute}
            for key_block in shape_keys.key_blocks
        },
    }


def _restore_source_runtime_state(source_obj, source_state):
    if source_state is None:
        return

    shape_keys = getattr(source_obj.data, "shape_keys", None)
    if shape_keys is None:
        return

    for key_block in shape_keys.key_blocks:
        key_state = source_state["key_values"].get(key_block.name)
        if key_state is None:
            continue
        key_block.value = key_state["value"]
        key_block.mute = key_state["mute"]

    source_obj.show_only_shape_key = source_state["show_only_shape_key"]
    source_obj.active_shape_key_index = min(
        max(0, source_state["active_shape_key_index"]),
        max(0, len(shape_keys.key_blocks) - 1),
    )


def _capture_target_runtime_state(target_obj):
    shape_keys = getattr(target_obj.data, "shape_keys", None)
    return {
        "active_shape_key_index": getattr(target_obj, "active_shape_key_index", 0),
        "show_only_shape_key": bool(getattr(target_obj, "show_only_shape_key", False)),
        "key_values": (
            {
                key_block.name: {"value": key_block.value, "mute": key_block.mute}
                for key_block in shape_keys.key_blocks
            }
            if shape_keys is not None
            else {}
        ),
    }


def _restore_target_runtime_state(target_obj, target_state, created_key_names):
    shape_keys = getattr(target_obj.data, "shape_keys", None)
    if shape_keys is not None:
        for key_block in shape_keys.key_blocks:
            if key_block.name in created_key_names:
                continue
            key_state = target_state["key_values"].get(key_block.name)
            if key_state is None:
                continue
            key_block.value = key_state["value"]
            key_block.mute = key_state["mute"]

    target_obj.show_only_shape_key = target_state["show_only_shape_key"]
    if shape_keys is not None and len(shape_keys.key_blocks) > 0:
        target_obj.active_shape_key_index = min(
            max(0, target_state["active_shape_key_index"]),
            len(shape_keys.key_blocks) - 1,
        )


def _set_source_preview_key(source_obj, key_index):
    shape_keys = getattr(source_obj.data, "shape_keys", None)
    if shape_keys is None or len(shape_keys.key_blocks) == 0:
        return

    for key_block in shape_keys.key_blocks:
        key_block.value = 0.0
    source_obj.show_only_shape_key = True
    source_obj.active_shape_key_index = min(max(0, key_index), len(shape_keys.key_blocks) - 1)


def _zero_target_shape_key_values(target_obj):
    shape_keys = getattr(target_obj.data, "shape_keys", None)
    if shape_keys is None:
        return

    for key_block in shape_keys.key_blocks:
        key_block.value = 0.0
    target_obj.show_only_shape_key = False


def _capture_evaluated_vertex_coordinates(obj, depsgraph):
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    try:
        return [vertex.co.copy() for vertex in evaluated_mesh.vertices]
    finally:
        evaluated_obj.to_mesh_clear()


def _run_object_operator(context, active_obj, operator_name, **kwargs):
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


def _bind_surface_deform_modifier(context, target_obj, modifier_name):
    result = _run_object_operator(
        context,
        target_obj,
        "surfacedeform_bind",
        modifier=modifier_name,
    )
    if "FINISHED" not in result:
        raise ValueError("Surface Deform bind did not finish. Check that the meshes overlap and try again.")


def transfer_shape_keys_by_wrap(context, source_obj, target_obj, replace_existing=True):
    """Bind the target to the source with Surface Deform and bake every source shape key."""
    source_shape_keys = getattr(source_obj.data, "shape_keys", None)
    if source_shape_keys is None or len(source_shape_keys.key_blocks) <= 1:
        raise ValueError("The source object has no transferable shape keys.")

    source_key_blocks = list(source_shape_keys.key_blocks)
    source_key_indices = {
        key_block.name: index
        for index, key_block in enumerate(source_key_blocks)
    }
    basis_key = source_key_blocks[0]
    use_relative = bool(getattr(source_shape_keys, "use_relative", True))

    target_basis_key = ensure_basis_shape_key(target_obj)
    target_basis_vertex_count = len(target_basis_key.data)

    source_state = _capture_source_runtime_state(source_obj)
    target_state = _capture_target_runtime_state(target_obj)
    resolved_target_keys = {basis_key.name: target_basis_key}
    created_keys = {}
    relative_names = {}
    created_count = 0
    replaced_count = 0
    skipped_count = 0
    pose_cache = {}
    temp_modifier = None

    def capture_target_pose(source_key):
        pose_name = source_key.name
        if pose_name in pose_cache:
            return pose_cache[pose_name]

        _set_source_preview_key(source_obj, source_key_indices[pose_name])
        _zero_target_shape_key_values(target_obj)
        context.view_layer.update()
        coordinates = _capture_evaluated_vertex_coordinates(target_obj, context.evaluated_depsgraph_get())
        if len(coordinates) != target_basis_vertex_count:
            raise ValueError(
                "Wrap Bake requires the target's evaluated mesh to keep the same vertex count. "
                "Disable topology-changing modifiers on the target first."
            )
        pose_cache[pose_name] = coordinates
        return coordinates

    def ensure_wrapped_key(source_key):
        nonlocal created_count, replaced_count, skipped_count

        if source_key.name == basis_key.name:
            return target_basis_key

        existing_resolved_key = resolved_target_keys.get(source_key.name)
        if existing_resolved_key is not None:
            return existing_resolved_key

        relative_source_key = (
            source_key.relative_key
            if use_relative and source_key.relative_key is not None
            else basis_key
        )
        target_relative_key = ensure_wrapped_key(relative_source_key)

        existing_target_key = None
        target_shape_keys = getattr(target_obj.data, "shape_keys", None)
        if target_shape_keys is not None:
            existing_target_key = target_shape_keys.key_blocks.get(source_key.name)

        if existing_target_key is not None:
            if replace_existing:
                if remove_shape_key_by_name(target_obj, source_key.name):
                    replaced_count += 1
            else:
                skipped_count += 1
                resolved_target_keys[source_key.name] = existing_target_key
                return existing_target_key

        reference_coordinates = capture_target_pose(relative_source_key)
        full_coordinates = capture_target_pose(source_key)
        target_relative_coordinates = [point.co.copy() for point in target_relative_key.data]

        target_key = target_obj.shape_key_add(name=source_key.name, from_mix=False)
        _copy_shape_key_metadata(source_key, target_key, value_override=0.0, mute_override=True)
        for vertex_index, target_relative_coordinate in enumerate(target_relative_coordinates):
            target_key.data[vertex_index].co = (
                target_relative_coordinate
                + (full_coordinates[vertex_index] - reference_coordinates[vertex_index])
            )

        resolved_target_keys[source_key.name] = target_key
        created_keys[source_key.name] = target_key
        relative_names[source_key.name] = relative_source_key.name if relative_source_key is not None else ""
        created_count += 1
        return target_key

    try:
        temp_modifier = target_obj.modifiers.new(_TEMP_SURFACE_DEFORM_NAME, "SURFACE_DEFORM")
        temp_modifier.target = source_obj

        _set_source_preview_key(source_obj, 0)
        _zero_target_shape_key_values(target_obj)
        context.view_layer.update()
        try:
            _bind_surface_deform_modifier(context, target_obj, temp_modifier.name)
        except RuntimeError as exc:
            raise ValueError(f"Surface Deform bind failed: {exc}") from exc

        for source_key in source_key_blocks[1:]:
            ensure_wrapped_key(source_key)
    finally:
        if temp_modifier is not None and target_obj.modifiers.get(temp_modifier.name) is not None:
            target_obj.modifiers.remove(temp_modifier)
        _restore_source_runtime_state(source_obj, source_state)
        _restore_target_runtime_state(target_obj, target_state, set(created_keys))
        context.view_layer.update()

    for source_key in source_key_blocks[1:]:
        target_key = created_keys.get(source_key.name)
        if target_key is None:
            continue
        _copy_shape_key_metadata(source_key, target_key)

    _apply_relative_keys(source_shape_keys, target_basis_key, resolved_target_keys, relative_names)
    return {
        "created_count": created_count,
        "replaced_count": replaced_count,
        "skipped_count": skipped_count,
    }
