"""Vertex-group mapping analysis helpers for Mod Tool."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .common import is_mesh_object, sync_edit_mesh_data


_WEIGHT_EPSILON = 1.0e-8
_TRIANGLE_EPSILON = 1.0e-12
_DIRECT_SCORE_THRESHOLD = 0.55
_DIRECT_MARGIN_THRESHOLD = 0.12
_UNMATCHED_SCORE_THRESHOLD = 0.30
_MIN_MATCHED_WEIGHT_RATIO = 0.35
_PARENT_BIAS_SCORE_DELTA = 0.08
_PARENT_BIAS_MIN_SCORE = 0.48


def resolve_group_mapping_objects(settings):
    """Resolve the explicit source and target objects used by group mapping."""
    source_obj = getattr(settings, "source_object", None)
    target_obj = getattr(settings, "target_object", None)
    if not is_mesh_object(source_obj):
        source_obj = None
    if not is_mesh_object(target_obj):
        target_obj = None
    if source_obj is not None and target_obj is not None and source_obj == target_obj:
        return source_obj, None
    return source_obj, target_obj


def _collect_vertex_group_data(obj, allowed_group_names=None):
    group_index_to_name = {group.index: group.name for group in obj.vertex_groups}
    allowed_names = None if allowed_group_names is None else set(allowed_group_names)
    vertex_weight_maps = []
    nonempty_names = set()
    group_weight_sum = defaultdict(float)
    group_weight_square_sum = defaultdict(float)
    group_weighted_local_sum = defaultdict(lambda: Vector((0.0, 0.0, 0.0)))

    for vertex in obj.data.vertices:
        vertex_weight_map = {}
        for assignment in vertex.groups:
            group_name = group_index_to_name.get(assignment.group)
            if not group_name:
                continue
            if allowed_names is not None and group_name not in allowed_names:
                continue
            if assignment.weight <= _WEIGHT_EPSILON:
                continue

            weight = float(assignment.weight)
            vertex_weight_map[group_name] = weight
            nonempty_names.add(group_name)
            group_weight_sum[group_name] += weight
            group_weight_square_sum[group_name] += weight * weight
            group_weighted_local_sum[group_name] += vertex.co * weight

        vertex_weight_maps.append(vertex_weight_map)

    ordered_names = [group.name for group in obj.vertex_groups if group.name in nonempty_names]
    return {
        "vertex_weight_maps": vertex_weight_maps,
        "group_names": ordered_names,
        "weight_sum": dict(group_weight_sum),
        "weight_square_sum": dict(group_weight_square_sum),
        "weighted_local_sum": dict(group_weighted_local_sum),
    }


def _resolve_deform_bone_map(obj):
    deform_bone_map = {}
    for modifier in obj.modifiers:
        if modifier.type != "ARMATURE" or modifier.object is None:
            continue
        if getattr(modifier.object, "type", None) != "ARMATURE":
            continue
        for bone in modifier.object.data.bones:
            if bone.use_deform:
                deform_bone_map[bone.name] = bone
    return deform_bone_map


def _resolve_candidate_group_names(target_obj, prefer_deform_groups):
    target_group_data = _collect_vertex_group_data(target_obj)
    all_nonempty_names = target_group_data["group_names"]
    if not prefer_deform_groups:
        return all_nonempty_names, _resolve_deform_bone_map(target_obj)

    deform_bone_map = _resolve_deform_bone_map(target_obj)
    deform_names = {name for name in deform_bone_map if name in set(all_nonempty_names)}
    if deform_names:
        return [name for name in all_nonempty_names if name in deform_names], deform_bone_map
    return all_nonempty_names, deform_bone_map


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
    return abs(denominator) <= (scale * _TRIANGLE_EPSILON)


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
    if abs(denominator) <= (scale * _TRIANGLE_EPSILON):
        return None

    weight_b = ((dot_ac_ac * dot_offset_ab) - (dot_ab_ac * dot_offset_ac)) / denominator
    weight_c = ((dot_ab_ab * dot_offset_ac) - (dot_ab_ac * dot_offset_ab)) / denominator
    weight_a = 1.0 - weight_b - weight_c
    return (weight_a, weight_b, weight_c)


def _build_surface_sampler(obj):
    sync_edit_mesh_data(obj)
    mesh = obj.data
    mesh.calc_loop_triangles()
    matrix_world = obj.matrix_world
    world_vertices = [matrix_world @ vertex.co for vertex in mesh.vertices]
    triangles = []
    ignored_triangle_count = 0

    for loop_triangle in mesh.loop_triangles:
        triangle = tuple(loop_triangle.vertices)
        tri_a, tri_b, tri_c = [world_vertices[index] for index in triangle]
        if _is_degenerate_triangle(tri_a, tri_b, tri_c):
            ignored_triangle_count += 1
            continue
        triangles.append(triangle)

    if not triangles:
        raise ValueError("The target mesh has no valid surface triangles to sample.")

    bvh_tree = BVHTree.FromPolygons(world_vertices, triangles, all_triangles=True)
    if bvh_tree is None:
        raise ValueError("Could not build a target surface sampler.")

    return {
        "world_vertices": world_vertices,
        "triangles": triangles,
        "bvh_tree": bvh_tree,
        "ignored_triangle_count": ignored_triangle_count,
    }


def _sample_target_weights_at_point(source_world_coordinate, sampler, target_vertex_weight_maps, tolerance):
    nearest = sampler["bvh_tree"].find_nearest(source_world_coordinate)
    if nearest is None:
        return None

    hit_coordinate, _normal, triangle_index, distance = nearest
    if triangle_index is None or distance is None or distance > tolerance:
        return None

    triangle = sampler["triangles"][triangle_index]
    tri_a, tri_b, tri_c = [sampler["world_vertices"][index] for index in triangle]
    weights = _barycentric_weights(hit_coordinate, tri_a, tri_b, tri_c)
    if weights is None:
        return None

    sampled_weights = defaultdict(float)
    for target_vertex_index, bary_weight in zip(triangle, weights):
        if bary_weight <= 0.0:
            continue
        for group_name, group_weight in target_vertex_weight_maps[target_vertex_index].items():
            sampled_weights[group_name] += group_weight * bary_weight

    return {
        "distance": float(distance),
        "weights": {
            group_name: weight
            for group_name, weight in sampled_weights.items()
            if weight > _WEIGHT_EPSILON
        },
    }


def _compute_bbox_diagonal(obj):
    corners = [Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(corner.x for corner in corners), min(corner.y for corner in corners), min(corner.z for corner in corners)))
    max_corner = Vector((max(corner.x for corner in corners), max(corner.y for corner in corners), max(corner.z for corner in corners)))
    diagonal = (max_corner - min_corner).length
    return diagonal if diagonal > _WEIGHT_EPSILON else 1.0


def _centroid_from_stats(stats, group_name):
    weight_sum = stats["weight_sum"].get(group_name, 0.0)
    if weight_sum <= _WEIGHT_EPSILON:
        return None
    return stats["weighted_local_sum"][group_name] / weight_sum


def _to_source_local(source_obj, target_obj, target_local_coordinate):
    target_world_coordinate = target_obj.matrix_world @ target_local_coordinate
    return source_obj.matrix_world.inverted() @ target_world_coordinate


def _centroid_score(source_centroid_local, target_centroid_local_in_source, bbox_diagonal):
    distance = (source_centroid_local - target_centroid_local_in_source).length
    normalized_distance = min(1.0, distance / max(bbox_diagonal, 1.0e-6))
    return max(0.0, 1.0 - normalized_distance), distance


def _side_label(local_x, bbox_diagonal):
    side_dead_zone = max(0.01 * bbox_diagonal, 1.0e-4)
    if local_x > side_dead_zone:
        return "POSITIVE"
    if local_x < -side_dead_zone:
        return "NEGATIVE"
    return "CENTER"


def _side_score(source_centroid_local, target_centroid_local_in_source, bbox_diagonal):
    source_side = _side_label(source_centroid_local.x, bbox_diagonal)
    target_side = _side_label(target_centroid_local_in_source.x, bbox_diagonal)
    if source_side == "CENTER" or target_side == "CENTER":
        return 0.5
    return 1.0 if source_side == target_side else 0.0


def _walk_bone_ancestors(bone):
    current = bone
    while current is not None:
        yield current
        current = current.parent


def _resolve_parent_bias(candidates, deform_bone_map):
    if len(candidates) < 2:
        return None, ""

    top_candidate = candidates[0]
    second_candidate = candidates[1]
    if abs(top_candidate["score"] - second_candidate["score"]) > _PARENT_BIAS_SCORE_DELTA:
        return None, ""

    top_bone = deform_bone_map.get(top_candidate["target_group"])
    second_bone = deform_bone_map.get(second_candidate["target_group"])
    if top_bone is None or second_bone is None:
        return None, ""

    top_ancestors = {bone.name for bone in _walk_bone_ancestors(top_bone)}
    second_ancestors = {bone.name for bone in _walk_bone_ancestors(second_bone)}

    if top_bone.name in second_ancestors and top_candidate["score"] >= _PARENT_BIAS_MIN_SCORE:
        return top_candidate, "Preferred parent/proximal target on the same bone chain."
    if second_bone.name in top_ancestors and second_candidate["score"] >= _PARENT_BIAS_MIN_SCORE:
        return second_candidate, "Preferred parent/proximal target on the same bone chain."
    return None, ""


def _build_candidate_rows(
    source_obj,
    target_obj,
    source_group_data,
    target_group_data,
    candidate_limit,
    overlap_numerators,
    target_norm_sums,
    matched_weight_sums,
    deform_bone_map,
):
    rows = []
    bbox_diagonal = _compute_bbox_diagonal(source_obj)

    for source_group_name in source_group_data["group_names"]:
        source_weight_sum = source_group_data["weight_sum"].get(source_group_name, 0.0)
        if source_weight_sum <= _WEIGHT_EPSILON:
            continue

        source_square_sum = source_group_data["weight_square_sum"].get(source_group_name, 0.0)
        source_centroid_local = _centroid_from_stats(source_group_data, source_group_name)
        matched_weight_ratio = matched_weight_sums.get(source_group_name, 0.0) / max(source_weight_sum, _WEIGHT_EPSILON)

        candidate_rows = []
        for target_group_name, numerator in overlap_numerators.get(source_group_name, {}).items():
            target_square_sum = target_norm_sums.get(target_group_name, 0.0)
            if numerator <= _WEIGHT_EPSILON or target_square_sum <= _WEIGHT_EPSILON:
                continue

            overlap_score = numerator / sqrt(max(source_square_sum * target_square_sum, _WEIGHT_EPSILON))
            target_centroid_local = _centroid_from_stats(target_group_data, target_group_name)
            if source_centroid_local is None or target_centroid_local is None:
                continue
            target_centroid_local_in_source = _to_source_local(source_obj, target_obj, target_centroid_local)
            centroid_score, centroid_distance = _centroid_score(
                source_centroid_local,
                target_centroid_local_in_source,
                bbox_diagonal,
            )
            side_score = _side_score(source_centroid_local, target_centroid_local_in_source, bbox_diagonal)
            total_score = (0.7 * overlap_score) + (0.2 * centroid_score) + (0.1 * side_score)
            confidence = total_score * min(1.0, matched_weight_ratio / max(_MIN_MATCHED_WEIGHT_RATIO, 1.0e-6))
            candidate_rows.append(
                {
                    "target_group": target_group_name,
                    "score": total_score,
                    "confidence": confidence,
                    "score_overlap": overlap_score,
                    "score_centroid": centroid_score,
                    "centroid_distance": centroid_distance,
                }
            )

        candidate_rows.sort(key=lambda item: (item["confidence"], item["score_overlap"], item["score_centroid"]), reverse=True)
        candidate_rows = candidate_rows[: max(1, candidate_limit)]

        if not candidate_rows or matched_weight_ratio < _MIN_MATCHED_WEIGHT_RATIO:
            rows.append(
                {
                    "enabled": False,
                    "source_group": source_group_name,
                    "target_group": candidate_rows[0]["target_group"] if candidate_rows else "",
                    "status": "UNMATCHED",
                    "confidence": candidate_rows[0]["confidence"] if candidate_rows else 0.0,
                    "score_overlap": candidate_rows[0]["score_overlap"] if candidate_rows else 0.0,
                    "score_centroid": candidate_rows[0]["score_centroid"] if candidate_rows else 0.0,
                    "reason": "Low surface coverage or no stable target weights were found.",
                }
            )
            continue

        preferred_candidate = candidate_rows[0]
        preferred_reason = "Strong overlap and centroid match."
        parent_biased_candidate, parent_reason = _resolve_parent_bias(candidate_rows, deform_bone_map)
        if parent_biased_candidate is not None:
            preferred_candidate = parent_biased_candidate
            preferred_reason = parent_reason

        second_best_score = 0.0
        for candidate_row in candidate_rows:
            if candidate_row["target_group"] != preferred_candidate["target_group"]:
                second_best_score = candidate_row["confidence"]
                break

        if preferred_candidate["confidence"] < _UNMATCHED_SCORE_THRESHOLD:
            status = "UNMATCHED"
            enabled = False
            reason = "The best target candidate scored too low."
        elif (
            preferred_candidate["confidence"] >= _DIRECT_SCORE_THRESHOLD
            and (
                second_best_score <= 0.0
                or (preferred_candidate["confidence"] - second_best_score) >= _DIRECT_MARGIN_THRESHOLD
                or parent_biased_candidate is not None
            )
        ):
            status = "DIRECT"
            enabled = True
            reason = preferred_reason
        else:
            status = "AMBIGUOUS"
            enabled = False
            competing_names = ", ".join(candidate["target_group"] for candidate in candidate_rows[:2] if candidate["target_group"])
            reason = f"Top candidates are too close: {competing_names}."

        rows.append(
            {
                "enabled": enabled,
                "source_group": source_group_name,
                "target_group": preferred_candidate["target_group"],
                "status": status,
                "confidence": preferred_candidate["confidence"],
                "score_overlap": preferred_candidate["score_overlap"],
                "score_centroid": preferred_candidate["score_centroid"],
                "reason": reason,
            }
        )

    return rows


def _resolve_conflicts(rows):
    direct_rows_by_target = defaultdict(list)
    for row in rows:
        if row["status"] == "DIRECT" and row["target_group"]:
            direct_rows_by_target[row["target_group"]].append(row)

    for target_group, conflicted_rows in direct_rows_by_target.items():
        if len(conflicted_rows) <= 1:
            continue

        conflicted_rows.sort(key=lambda item: item["confidence"], reverse=True)
        keeper = conflicted_rows[0]
        keeper["reason"] = keeper["reason"] if keeper["reason"] else "Kept strongest direct match."

        for conflicted_row in conflicted_rows[1:]:
            conflicted_row["status"] = "CONFLICT"
            conflicted_row["enabled"] = False
            conflicted_row["reason"] = f"Target group '{target_group}' is claimed by a stronger source match."


def store_group_match_rows(scene, rows):
    """Write vertex-group mapping rows back to the Blender scene collection."""
    scene.mdt_group_match_results.clear()
    for row in rows:
        item = scene.mdt_group_match_results.add()
        item.source_group = row["source_group"]
        item.target_group = row["target_group"]
        item.status = row["status"]
        item.confidence = row["confidence"]
        item.score_overlap = row["score_overlap"]
        item.score_centroid = row["score_centroid"]
        item.reason = row["reason"]

    if len(scene.mdt_group_match_results) == 0:
        scene.mdt_active_group_match_result_index = 0
    else:
        scene.mdt_active_group_match_result_index = min(
            scene.mdt_active_group_match_result_index,
            len(scene.mdt_group_match_results) - 1,
        )


def count_group_match_status(scene, status):
    """Count how many stored vertex-group mappings share the requested status."""
    return sum(1 for item in scene.mdt_group_match_results if item.status == status)


def count_applicable_group_matches(scene):
    """Count rows that can currently be applied as renames."""
    return sum(
        1
        for item in scene.mdt_group_match_results
        if item.status in {"DIRECT", "MANUAL"} and item.source_group and item.target_group
    )


def analyze_group_mapping(source_obj, target_obj, candidate_limit=3, surface_tolerance=0.001, prefer_deform_groups=True):
    """Analyze source-to-target vertex-group rename suggestions from aligned meshes."""
    if not is_mesh_object(source_obj) or not is_mesh_object(target_obj):
        raise ValueError("Choose valid mesh objects for Source and Target.")
    if source_obj == target_obj:
        raise ValueError("Source and Target must be different objects.")
    if candidate_limit < 1:
        raise ValueError("Candidate Limit must be at least 1.")
    if surface_tolerance < 0.0:
        raise ValueError("Surface Tolerance must be zero or greater.")

    sync_edit_mesh_data(source_obj)
    sync_edit_mesh_data(target_obj)

    source_group_data = _collect_vertex_group_data(source_obj)
    if len(source_group_data["group_names"]) == 0:
        raise ValueError("The source object has no non-empty vertex groups.")

    candidate_group_names, deform_bone_map = _resolve_candidate_group_names(target_obj, prefer_deform_groups)
    if len(candidate_group_names) == 0:
        raise ValueError("The target object has no non-empty candidate vertex groups.")

    target_group_data = _collect_vertex_group_data(target_obj, candidate_group_names)
    if len(target_group_data["group_names"]) == 0:
        raise ValueError("The target object has no usable vertex groups after filtering.")

    sampler = _build_surface_sampler(target_obj)
    source_world_vertices = [source_obj.matrix_world @ vertex.co for vertex in source_obj.data.vertices]
    overlap_numerators = defaultdict(lambda: defaultdict(float))
    target_norm_sums = defaultdict(float)
    matched_weight_sums = defaultdict(float)
    unmatched_vertex_count = 0
    matched_vertex_count = 0
    farthest_distance = 0.0

    for vertex_index, source_world_coordinate in enumerate(source_world_vertices):
        sampled_payload = _sample_target_weights_at_point(
            source_world_coordinate,
            sampler,
            target_group_data["vertex_weight_maps"],
            surface_tolerance,
        )
        source_weights = source_group_data["vertex_weight_maps"][vertex_index]
        if sampled_payload is None:
            unmatched_vertex_count += 1
            continue

        matched_vertex_count += 1
        farthest_distance = max(farthest_distance, sampled_payload["distance"])
        sampled_target_weights = sampled_payload["weights"]
        for target_group_name, sampled_weight in sampled_target_weights.items():
            target_norm_sums[target_group_name] += sampled_weight * sampled_weight

        for source_group_name, source_weight in source_weights.items():
            matched_weight_sums[source_group_name] += source_weight
            if not sampled_target_weights:
                continue
            numerators = overlap_numerators[source_group_name]
            for target_group_name, sampled_weight in sampled_target_weights.items():
                numerators[target_group_name] += source_weight * sampled_weight

    rows = _build_candidate_rows(
        source_obj,
        target_obj,
        source_group_data,
        target_group_data,
        candidate_limit,
        overlap_numerators,
        target_norm_sums,
        matched_weight_sums,
        deform_bone_map,
    )
    _resolve_conflicts(rows)

    return {
        "rows": rows,
        "candidate_group_count": len(target_group_data["group_names"]),
        "source_group_count": len(source_group_data["group_names"]),
        "matched_vertex_count": matched_vertex_count,
        "unmatched_vertex_count": unmatched_vertex_count,
        "farthest_distance": farthest_distance,
        "ignored_triangle_count": sampler["ignored_triangle_count"],
        "used_deform_filter": bool(prefer_deform_groups and len(deform_bone_map) > 0),
    }


def apply_group_mapping_rename(source_obj, mapping_items):
    """Apply exact target names from the stored mapping table without suffixing conflicts."""
    if not is_mesh_object(source_obj):
        raise ValueError("Choose a valid source mesh.")

    source_groups_by_name = {group.name: group for group in source_obj.vertex_groups}
    raw_pairs = []
    skipped_missing = 0

    for item in mapping_items:
        if item.status not in {"DIRECT", "MANUAL"} or not item.source_group or not item.target_group:
            continue
        source_group = source_groups_by_name.get(item.source_group)
        if source_group is None:
            skipped_missing += 1
            continue
        if source_group.name == item.target_group:
            continue
        raw_pairs.append((source_group, item.target_group))

    if not raw_pairs:
        return {
            "renamed_count": 0,
            "skipped_conflict_count": 0,
            "skipped_missing_count": skipped_missing,
        }

    rename_source_names = {group.name for group, _target_name in raw_pairs}
    existing_names = {group.name for group in source_obj.vertex_groups}
    used_target_names = set()
    resolved_pairs = []
    skipped_conflict_count = 0

    for group, target_name in raw_pairs:
        if target_name in used_target_names:
            skipped_conflict_count += 1
            continue
        if target_name in existing_names and target_name not in rename_source_names:
            skipped_conflict_count += 1
            continue
        used_target_names.add(target_name)
        resolved_pairs.append((group, target_name))

    if not resolved_pairs:
        return {
            "renamed_count": 0,
            "skipped_conflict_count": skipped_conflict_count,
            "skipped_missing_count": skipped_missing,
        }

    from .rename import apply_rename_plan

    renamed_count = apply_rename_plan(resolved_pairs)
    return {
        "renamed_count": renamed_count,
        "skipped_conflict_count": skipped_conflict_count,
        "skipped_missing_count": skipped_missing,
    }
