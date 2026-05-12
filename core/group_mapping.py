"""Vertex-group spatial mapping helpers for Mod Tool."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
import re

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree

from .common import get_world_point_coordinates, is_mesh_object, sync_edit_mesh_data


_WEIGHT_EPSILON = 1.0e-8
_POINT_WEIGHT_FLOOR = 1.0e-4
_MAX_POINT_WEIGHTS = 4
_MAX_NEIGHBOR_POINTS = 8
_MAX_CANDIDATES_PER_SOURCE = 8

_MATCH_RADIUS_RATIO = 0.018
_MIN_MATCH_RADIUS = 1.0e-5

_DIRECT_SCORE_THRESHOLD = 0.58
_DIRECT_MARGIN_THRESHOLD = 0.08
_MERGE_SCORE_THRESHOLD = 0.48
_MERGE_MARGIN_THRESHOLD = 0.07

_APPLICABLE_STATUSES = {"DIRECT", "MERGE", "FALLBACK"}


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


def _resolve_deform_group_names(obj):
    deform_bone_map = _resolve_deform_bone_map(obj)
    deform_names = set(deform_bone_map)
    return [group.name for group in obj.vertex_groups if group.name in deform_names], deform_bone_map


def _resolve_target_candidate_group_names(target_obj):
    deform_group_names, _deform_bone_map = _resolve_deform_group_names(target_obj)
    if deform_group_names:
        return deform_group_names
    return [group.name for group in target_obj.vertex_groups]


def _capture_world_positions(obj, depsgraph):
    if depsgraph is not None:
        evaluated_obj = obj.evaluated_get(depsgraph)
        evaluated_mesh = bpy.data.meshes.new_from_object(
            evaluated_obj,
            depsgraph=depsgraph,
            preserve_all_data_layers=False,
        )
        try:
            if len(evaluated_mesh.vertices) == len(obj.data.vertices):
                return get_world_point_coordinates(evaluated_mesh.vertices, evaluated_obj.matrix_world), True
        finally:
            bpy.data.meshes.remove(evaluated_mesh)

    return get_world_point_coordinates(obj.data.vertices, obj.matrix_world), False


def _collect_vertex_group_data(obj, allowed_group_names=None, world_positions=None):
    group_index_to_name = {group.index: group.name for group in obj.vertex_groups}
    allowed_names = None if allowed_group_names is None else set(allowed_group_names)
    vertex_weight_maps = []
    nonempty_names = set()
    group_weight_sum = defaultdict(float)
    group_weight_square_sum = defaultdict(float)
    group_weighted_world_sum = defaultdict(lambda: Vector((0.0, 0.0, 0.0)))

    if world_positions is None:
        world_positions = get_world_point_coordinates(obj.data.vertices, obj.matrix_world)

    for vertex in obj.data.vertices:
        vertex_weight_map = {}
        if vertex.index < len(world_positions):
            world_position = world_positions[vertex.index]
        else:
            world_position = obj.matrix_world @ vertex.co

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
            group_weighted_world_sum[group_name] += world_position * weight

        vertex_weight_maps.append(vertex_weight_map)

    if allowed_names is None:
        ordered_names = [group.name for group in obj.vertex_groups if group.name in nonempty_names]
    else:
        ordered_names = [group.name for group in obj.vertex_groups if group.name in allowed_names]
    return {
        "vertex_weight_maps": vertex_weight_maps,
        "group_names": ordered_names,
        "nonempty_group_names": [group.name for group in obj.vertex_groups if group.name in nonempty_names],
        "weight_sum": dict(group_weight_sum),
        "weight_square_sum": dict(group_weight_square_sum),
        "weighted_world_sum": dict(group_weighted_world_sum),
        "world_positions": world_positions,
    }


def _sorted_point_weights(weight_map):
    items = [
        (group_name, float(weight))
        for group_name, weight in weight_map.items()
        if float(weight) > _POINT_WEIGHT_FLOOR
    ]
    if not items:
        return []

    items.sort(key=lambda item: (-item[1], item[0]))
    items = items[:_MAX_POINT_WEIGHTS]
    total = sum(weight for _group_name, weight in items)
    if total <= _WEIGHT_EPSILON:
        return []
    return [(group_name, weight / total) for group_name, weight in items]


def _build_weighted_points(group_data):
    points = []
    world_positions = group_data["world_positions"]
    for vertex_index, weight_map in enumerate(group_data["vertex_weight_maps"]):
        weights = _sorted_point_weights(weight_map)
        if not weights:
            continue
        if vertex_index >= len(world_positions):
            continue
        points.append((world_positions[vertex_index], weights))
    return points


def _build_kdtree(points):
    kd_tree = KDTree(len(points))
    for point_index, (world_position, _weights) in enumerate(points):
        kd_tree.insert(world_position, point_index)
    kd_tree.balance()
    return kd_tree


def _positions_diagonal(positions):
    if not positions:
        return 0.0
    min_x = min(position.x for position in positions)
    min_y = min(position.y for position in positions)
    min_z = min(position.z for position in positions)
    max_x = max(position.x for position in positions)
    max_y = max(position.y for position in positions)
    max_z = max(position.z for position in positions)
    return sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)


def _positions_center(positions):
    if not positions:
        return Vector((0.0, 0.0, 0.0))
    min_x = min(position.x for position in positions)
    min_y = min(position.y for position in positions)
    min_z = min(position.z for position in positions)
    max_x = max(position.x for position in positions)
    max_y = max(position.y for position in positions)
    max_z = max(position.z for position in positions)
    return Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5))


def _estimate_match_radius(source_points, target_points):
    sample_positions = [point[0] for point in source_points[:20000]]
    sample_positions.extend(point[0] for point in target_points[:20000])
    diagonal = _positions_diagonal(sample_positions)
    return max(diagonal * _MATCH_RADIUS_RATIO, _MIN_MATCH_RADIUS)


def _compute_point_cloud_votes(source_points, target_points, radius):
    target_tree = _build_kdtree(target_points)
    source_mass = defaultdict(float)
    source_hit_mass = defaultdict(float)
    pair_votes = defaultdict(lambda: defaultdict(float))
    matched_point_count = 0

    for source_position, source_weights in source_points:
        for source_group, source_weight in source_weights:
            source_mass[source_group] += source_weight

        neighbors = target_tree.find_n(source_position, _MAX_NEIGHBOR_POINTS)
        if not neighbors:
            continue

        nearest_distance = max(float(neighbors[0][2]), 0.0)
        hit_quality = min(1.0, radius / max(nearest_distance, radius))
        if hit_quality <= _WEIGHT_EPSILON:
            continue

        local_target_scores = defaultdict(float)
        for _coordinate, target_index, distance in neighbors:
            distance_score = 1.0 / (1.0 + float(distance) / max(radius, _MIN_MATCH_RADIUS))
            for target_group, target_weight in target_points[target_index][1]:
                local_target_scores[target_group] += target_weight * distance_score

        local_total = sum(local_target_scores.values())
        if local_total <= _WEIGHT_EPSILON:
            continue

        matched_point_count += 1
        for source_group, source_weight in source_weights:
            weighted_hit = source_weight * hit_quality
            source_hit_mass[source_group] += weighted_hit
            numerators = pair_votes[source_group]
            for target_group, target_score in local_target_scores.items():
                numerators[target_group] += weighted_hit * (target_score / local_total)

    scores = {}
    for source_group, target_scores in pair_votes.items():
        source_total = source_mass.get(source_group, 0.0)
        if source_total <= _WEIGHT_EPSILON:
            continue
        scores[source_group] = {
            target_group: min(1.0, score / source_total)
            for target_group, score in target_scores.items()
            if score > _WEIGHT_EPSILON
        }

    coverage = {
        source_group: min(1.0, source_hit_mass.get(source_group, 0.0) / max(total, _WEIGHT_EPSILON))
        for source_group, total in source_mass.items()
    }
    return {
        "scores": scores,
        "coverage": coverage,
        "matched_point_count": matched_point_count,
    }


def _centroid_from_stats(stats, group_name):
    weight_sum = stats["weight_sum"].get(group_name, 0.0)
    if weight_sum <= _WEIGHT_EPSILON:
        return None
    return stats["weighted_world_sum"][group_name] / weight_sum


def _centroid_score(source_centroid, target_centroid, bbox_diagonal):
    if source_centroid is None or target_centroid is None:
        return 0.0, 0.0
    distance = (source_centroid - target_centroid).length
    normalized_distance = min(1.0, distance / max(bbox_diagonal, _MIN_MATCH_RADIUS))
    return max(0.0, 1.0 - normalized_distance), distance


def _side_label(world_x, center_x, bbox_diagonal):
    side_dead_zone = max(0.01 * bbox_diagonal, 1.0e-4)
    offset = world_x - center_x
    if offset > side_dead_zone:
        return "POSITIVE"
    if offset < -side_dead_zone:
        return "NEGATIVE"
    return "CENTER"


def _side_score(source_centroid, target_centroid, bbox_center, bbox_diagonal):
    if source_centroid is None or target_centroid is None:
        return 0.0
    source_side = _side_label(source_centroid.x, bbox_center.x, bbox_diagonal)
    target_side = _side_label(target_centroid.x, bbox_center.x, bbox_diagonal)
    if source_side == "CENTER" or target_side == "CENTER":
        return 0.5
    return 1.0 if source_side == target_side else 0.0


def _normalized_group_name(name):
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def _name_score(source_group, target_group):
    if source_group == target_group:
        return 1.0
    source_normalized = _normalized_group_name(source_group)
    target_normalized = _normalized_group_name(target_group)
    if source_normalized and source_normalized == target_normalized:
        return 0.85
    return 0.0


def _candidate_score(cloud_score, centroid_value, side_value, name_value):
    total_score = (
        0.82 * cloud_score
        + 0.12 * centroid_value
        + 0.04 * side_value
        + 0.02 * name_value
    )
    if cloud_score <= 0.02:
        total_score = min(total_score, 0.32 + 0.08 * name_value)
    return min(1.0, max(0.0, total_score))


def _build_candidate(
    source_group,
    target_group,
    source_group_data,
    target_group_data,
    vote_scores,
    vote_coverage,
    bbox_center,
    bbox_diagonal,
):
    source_centroid = _centroid_from_stats(source_group_data, source_group)
    target_centroid = _centroid_from_stats(target_group_data, target_group)
    centroid_value, centroid_distance = _centroid_score(source_centroid, target_centroid, bbox_diagonal)
    side_value = _side_score(source_centroid, target_centroid, bbox_center, bbox_diagonal)
    cloud_score = vote_scores.get(source_group, {}).get(target_group, 0.0)
    name_value = _name_score(source_group, target_group)
    confidence = _candidate_score(cloud_score, centroid_value, side_value, name_value)
    confidence *= max(vote_coverage.get(source_group, 0.0), 0.20 if name_value > 0.0 else 0.0)

    return {
        "source_group": source_group,
        "target_group": target_group,
        "confidence": confidence,
        "score_overlap": cloud_score,
        "score_centroid": centroid_value,
        "centroid_distance": centroid_distance,
        "score_name": name_value,
    }


def _build_candidates_by_source(
    source_group_data,
    target_group_data,
    target_group_names,
    vote_scores,
    vote_coverage,
):
    target_group_name_set = set(target_group_names)
    source_positions = source_group_data["world_positions"]
    target_positions = target_group_data["world_positions"]
    bbox_positions = source_positions[:20000] + target_positions[:20000]
    bbox_diagonal = max(_positions_diagonal(bbox_positions), _MIN_MATCH_RADIUS)
    bbox_center = _positions_center(bbox_positions)

    normalized_targets = defaultdict(list)
    for target_group in target_group_names:
        normalized_targets[_normalized_group_name(target_group)].append(target_group)

    candidates_by_source = {}
    for source_group in source_group_data["group_names"]:
        candidate_target_names = set(vote_scores.get(source_group, {}))
        if source_group in target_group_name_set:
            candidate_target_names.add(source_group)
        candidate_target_names.update(normalized_targets.get(_normalized_group_name(source_group), ()))

        candidates = [
            _build_candidate(
                source_group,
                target_group,
                source_group_data,
                target_group_data,
                vote_scores,
                vote_coverage,
                bbox_center,
                bbox_diagonal,
            )
            for target_group in candidate_target_names
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate["confidence"] > _WEIGHT_EPSILON
            or candidate["score_overlap"] > _WEIGHT_EPSILON
            or candidate["score_name"] > 0.0
        ]
        candidates.sort(
            key=lambda item: (
                item["confidence"],
                item["score_overlap"],
                item["score_centroid"],
                item["score_name"],
            ),
            reverse=True,
        )
        candidates_by_source[source_group] = candidates[:_MAX_CANDIDATES_PER_SOURCE]

    return candidates_by_source, bbox_center, bbox_diagonal


def _row_from_candidate(source_group, candidate, status, reason):
    return {
        "source_group": source_group,
        "target_group": candidate["target_group"],
        "status": status,
        "confidence": candidate["confidence"],
        "score_overlap": candidate["score_overlap"],
        "score_centroid": candidate["score_centroid"],
        "reason": reason,
    }


def _find_unused_candidate(candidates, claimed_targets):
    for candidate in candidates:
        if not claimed_targets.get(candidate["target_group"]):
            return candidate
    return None


def _should_merge(best_candidate, best_unused_candidate):
    if best_candidate["confidence"] < _MERGE_SCORE_THRESHOLD:
        return best_unused_candidate is None
    if best_unused_candidate is None:
        return True
    return best_candidate["confidence"] >= best_unused_candidate["confidence"] + _MERGE_MARGIN_THRESHOLD


def _fallback_candidate(
    source_group,
    source_group_data,
    target_group_names,
    target_group_data,
    claimed_targets,
    candidates,
    bbox_diagonal,
):
    if source_group in target_group_names:
        return {
            "source_group": source_group,
            "target_group": source_group,
            "confidence": 0.28,
            "score_overlap": 0.0,
            "score_centroid": 0.0,
            "centroid_distance": 0.0,
            "score_name": 1.0,
            "reason": "Fallback kept the exact same group name.",
        }

    source_normalized = _normalized_group_name(source_group)
    for target_group in target_group_names:
        if source_normalized and source_normalized == _normalized_group_name(target_group):
            return {
                "source_group": source_group,
                "target_group": target_group,
                "confidence": 0.24,
                "score_overlap": 0.0,
                "score_centroid": 0.0,
                "centroid_distance": 0.0,
                "score_name": 0.85,
                "reason": "Fallback matched the normalized group name.",
            }

    if candidates:
        candidate = candidates[0].copy()
        candidate["reason"] = "Fallback used the strongest available point-cloud candidate."
        return candidate

    source_centroid = _centroid_from_stats(source_group_data, source_group)
    target_centroids = []
    for target_group in target_group_data["group_names"]:
        target_centroid = _centroid_from_stats(target_group_data, target_group)
        if target_centroid is not None:
            target_centroids.append((target_group, target_centroid))

    if source_centroid is not None and target_centroids:
        unclaimed_centroids = [
            (target_group, centroid)
            for target_group, centroid in target_centroids
            if not claimed_targets.get(target_group)
        ]
        centroid_pool = unclaimed_centroids if unclaimed_centroids else target_centroids
        target_group, target_centroid = min(
            centroid_pool,
            key=lambda item: (source_centroid - item[1]).length_squared,
        )
        centroid_value, centroid_distance = _centroid_score(source_centroid, target_centroid, bbox_diagonal)
        return {
            "source_group": source_group,
            "target_group": target_group,
            "confidence": min(0.22, centroid_value * 0.22),
            "score_overlap": 0.0,
            "score_centroid": centroid_value,
            "centroid_distance": centroid_distance,
            "score_name": 0.0,
            "reason": "Fallback used the nearest target weight centroid.",
        }

    unclaimed_targets = [target_group for target_group in target_group_names if not claimed_targets.get(target_group)]
    target_group = (unclaimed_targets or target_group_names)[0]
    return {
        "source_group": source_group,
        "target_group": target_group,
        "confidence": 0.10,
        "score_overlap": 0.0,
        "score_centroid": 0.0,
        "centroid_distance": 0.0,
        "score_name": 0.0,
        "reason": "Fallback used the first available target group.",
    }


def _assign_mapping_rows(
    source_group_data,
    target_group_data,
    target_group_names,
    candidates_by_source,
    bbox_diagonal,
):
    assigned_rows = {}
    claimed_targets = defaultdict(list)
    anchor_candidates = []
    source_order = list(source_group_data["group_names"])

    for source_group in source_order:
        candidates = candidates_by_source.get(source_group, [])
        if not candidates:
            continue
        best_candidate = candidates[0]
        second_score = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
        margin = best_candidate["confidence"] - second_score
        if best_candidate["confidence"] >= _DIRECT_SCORE_THRESHOLD and margin >= _DIRECT_MARGIN_THRESHOLD:
            anchor_candidates.append((best_candidate["confidence"], margin, source_group, best_candidate))

    anchor_candidates.sort(reverse=True)
    for _confidence, _margin, source_group, candidate in anchor_candidates:
        target_group = candidate["target_group"]
        if source_group in assigned_rows or claimed_targets.get(target_group):
            continue
        assigned_rows[source_group] = _row_from_candidate(
            source_group,
            candidate,
            "DIRECT",
            "High-confidence one-to-one point-cloud match.",
        )
        claimed_targets[target_group].append(source_group)

    remaining_sources = [source_group for source_group in source_order if source_group not in assigned_rows]
    remaining_sources.sort(
        key=lambda source_group: (
            candidates_by_source.get(source_group, [{}])[0].get("confidence", 0.0),
            candidates_by_source.get(source_group, [{}])[0].get("score_overlap", 0.0),
        ),
        reverse=True,
    )

    for source_group in remaining_sources:
        candidates = candidates_by_source.get(source_group, [])
        if not candidates:
            candidate = _fallback_candidate(
                source_group,
                source_group_data,
                target_group_names,
                target_group_data,
                claimed_targets,
                candidates,
                bbox_diagonal,
            )
            status = "FALLBACK"
            reason = candidate.get("reason", "Fallback kept this source deform group mapped.")
        else:
            best_candidate = candidates[0]
            best_unused_candidate = _find_unused_candidate(candidates, claimed_targets)
            target_is_claimed = bool(claimed_targets.get(best_candidate["target_group"]))

            if not target_is_claimed:
                candidate = best_candidate
                status = "DIRECT" if best_candidate["confidence"] >= _DIRECT_SCORE_THRESHOLD else "FALLBACK"
                reason = (
                    "Point-cloud match selected an unused target group."
                    if status == "DIRECT"
                    else "Fallback used the best unused point-cloud candidate."
                )
            elif _should_merge(best_candidate, best_unused_candidate):
                candidate = best_candidate
                status = "MERGE" if best_candidate["confidence"] >= _MERGE_SCORE_THRESHOLD else "FALLBACK"
                reason = (
                    "Best spatial candidate reuses an already claimed target, so the source group will merge into it."
                    if status == "MERGE"
                    else "Fallback reused the closest claimed target to keep this source group mapped."
                )
            elif best_unused_candidate is not None:
                candidate = best_unused_candidate
                status = "DIRECT" if best_unused_candidate["confidence"] >= _DIRECT_SCORE_THRESHOLD else "FALLBACK"
                reason = (
                    "Point-cloud match chose the best unused target after a stronger target was already claimed."
                    if status == "DIRECT"
                    else "Fallback chose the best unused target after the top target was already claimed."
                )
            else:
                candidate = _fallback_candidate(
                    source_group,
                    source_group_data,
                    target_group_names,
                    target_group_data,
                    claimed_targets,
                    candidates,
                    bbox_diagonal,
                )
                status = "FALLBACK"
                reason = candidate.get("reason", "Fallback kept this source deform group mapped.")

        assigned_rows[source_group] = _row_from_candidate(source_group, candidate, status, reason)
        claimed_targets[candidate["target_group"]].append(source_group)

    return [assigned_rows[source_group] for source_group in source_order]


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


def count_group_match_statuses(scene):
    """Count stored vertex-group mapping statuses in one pass."""
    counts = {"DIRECT": 0, "MERGE": 0, "FALLBACK": 0}
    for item in scene.mdt_group_match_results:
        if item.status in counts:
            counts[item.status] += 1
    return counts


def analyze_group_mapping(source_obj, target_obj, depsgraph=None):
    """Analyze source-to-target vertex-group mapping from spatial weight clouds."""
    if not is_mesh_object(source_obj) or not is_mesh_object(target_obj):
        raise ValueError("Choose valid mesh objects for Source and Target.")
    if source_obj == target_obj:
        raise ValueError("Source and Target must be different objects.")

    sync_edit_mesh_data(source_obj)
    sync_edit_mesh_data(target_obj)

    source_deform_names, source_deform_bone_map = _resolve_deform_group_names(source_obj)
    if len(source_deform_names) == 0 or len(source_deform_bone_map) == 0:
        raise ValueError("The source object needs an Armature modifier with deform-bone vertex groups.")

    target_group_names = _resolve_target_candidate_group_names(target_obj)
    if len(target_group_names) == 0:
        raise ValueError("The target object has no vertex groups to use as rename targets.")

    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    source_positions, source_used_evaluated_positions = _capture_world_positions(source_obj, depsgraph)
    target_positions, target_used_evaluated_positions = _capture_world_positions(target_obj, depsgraph)

    source_group_data = _collect_vertex_group_data(source_obj, source_deform_names, source_positions)
    if len(source_group_data["group_names"]) == 0:
        raise ValueError("The source object has no deform-bone vertex groups.")

    target_group_data = _collect_vertex_group_data(target_obj, target_group_names, target_positions)
    if len(target_group_data["nonempty_group_names"]) == 0:
        raise ValueError("The target object has no non-empty vertex groups for spatial matching.")

    source_points = _build_weighted_points(source_group_data)
    target_points = _build_weighted_points(target_group_data)
    if not source_points:
        raise ValueError("The source object has no weighted deform points to match.")
    if not target_points:
        raise ValueError("The target object has no weighted points to match.")

    match_radius = _estimate_match_radius(source_points, target_points)
    vote_data = _compute_point_cloud_votes(source_points, target_points, match_radius)
    candidates_by_source, _bbox_center, bbox_diagonal = _build_candidates_by_source(
        source_group_data,
        target_group_data,
        target_group_names,
        vote_data["scores"],
        vote_data["coverage"],
    )
    rows = _assign_mapping_rows(
        source_group_data,
        target_group_data,
        target_group_names,
        candidates_by_source,
        bbox_diagonal,
    )

    return {
        "rows": rows,
        "candidate_group_count": len(target_group_data["group_names"]),
        "source_group_count": len(source_group_data["group_names"]),
        "source_deform_group_count": len(source_deform_names),
        "target_group_count": len(target_group_names),
        "source_point_count": len(source_points),
        "target_point_count": len(target_points),
        "matched_point_count": vote_data["matched_point_count"],
        "match_radius": match_radius,
        "source_used_evaluated_positions": source_used_evaluated_positions,
        "target_used_evaluated_positions": target_used_evaluated_positions,
    }


def _capture_group_weights(obj):
    group_index_to_name = {group.index: group.name for group in obj.vertex_groups}
    weights_by_name = defaultdict(dict)

    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.weight <= _WEIGHT_EPSILON:
                continue
            group_name = group_index_to_name.get(assignment.group)
            if not group_name:
                continue
            weights_by_name[group_name][vertex.index] = float(assignment.weight)

    return dict(weights_by_name)


def _resolve_target_valid_group_names(target_obj):
    return [group.name for group in target_obj.vertex_groups]


def _ordered_target_names(target_obj, mapped_target_names):
    mapped_names = set(mapped_target_names)
    target_names = _resolve_target_valid_group_names(target_obj)
    target_order = [name for name in target_names if name in mapped_names]
    target_order.extend(name for name in mapped_target_names if name not in set(target_order))
    return target_order, set(target_names)


def _add_sparse_group_weights(obj, group_name, weights_by_vertex_index):
    vertex_group = obj.vertex_groups.new(name=group_name)
    for vertex_index, weight in weights_by_vertex_index.items():
        if weight > _WEIGHT_EPSILON:
            vertex_group.add([vertex_index], min(1.0, weight), "REPLACE")
    return vertex_group


def apply_group_mapping_rename(source_obj, target_obj, mapping_items):
    """Rebuild source vertex groups from applicable mapping rows and delete helper groups."""
    if not is_mesh_object(source_obj):
        raise ValueError("Choose a valid source mesh.")
    if not is_mesh_object(target_obj):
        raise ValueError("Choose a valid target mesh.")

    sync_edit_mesh_data(source_obj)
    original_group_names = [group.name for group in source_obj.vertex_groups]
    original_weights_by_name = _capture_group_weights(source_obj)
    source_deform_bone_map = _resolve_deform_bone_map(source_obj)
    source_deform_names = {
        name
        for name in original_group_names
        if name in source_deform_bone_map
    }
    for source_name in source_deform_names:
        original_weights_by_name.setdefault(source_name, {})

    mapping_by_target = defaultdict(list)
    used_sources = {}
    skipped_missing_count = 0
    skipped_conflict_count = 0

    _target_deform_order, valid_target_names = _ordered_target_names(target_obj, [])
    if not valid_target_names:
        raise ValueError("The target object has no vertex groups to use as rename targets.")

    for item in mapping_items:
        if item.status not in _APPLICABLE_STATUSES:
            continue

        source_name = item.source_group
        target_name = item.target_group
        if not source_name or not target_name:
            skipped_missing_count += 1
            continue
        if source_name not in original_weights_by_name:
            skipped_missing_count += 1
            continue
        if target_name not in valid_target_names:
            skipped_conflict_count += 1
            continue

        previous_target = used_sources.get(source_name)
        if previous_target is not None and previous_target != target_name:
            skipped_conflict_count += 1
            continue

        used_sources[source_name] = target_name
        if source_name not in mapping_by_target[target_name]:
            mapping_by_target[target_name].append(source_name)

    if not mapping_by_target:
        return {
            "created_count": 0,
            "mapped_source_count": 0,
            "merged_count": 0,
            "removed_non_deform_count": 0,
            "removed_unmapped_deform_count": 0,
            "skipped_conflict_count": skipped_conflict_count,
            "skipped_missing_count": skipped_missing_count,
        }

    target_order, _valid_target_names = _ordered_target_names(target_obj, mapping_by_target.keys())
    target_order = [name for name in target_order if name in mapping_by_target]

    combined_weights_by_target = {}
    for target_name in target_order:
        combined_weights = defaultdict(float)
        for source_name in mapping_by_target[target_name]:
            for vertex_index, weight in original_weights_by_name[source_name].items():
                combined_weights[vertex_index] = min(1.0, combined_weights[vertex_index] + weight)
        combined_weights_by_target[target_name] = dict(combined_weights)

    for group in reversed(list(source_obj.vertex_groups)):
        source_obj.vertex_groups.remove(group)

    created_count = 0
    for target_name in target_order:
        weights_by_vertex_index = combined_weights_by_target[target_name]
        _add_sparse_group_weights(source_obj, target_name, weights_by_vertex_index)
        created_count += 1

    mapped_source_names = set(used_sources)
    if source_deform_names:
        removed_non_deform_count = sum(1 for name in original_group_names if name not in source_deform_names)
        removed_unmapped_deform_count = sum(
            1 for name in original_group_names if name in source_deform_names and name not in mapped_source_names
        )
    else:
        removed_non_deform_count = len([name for name in original_group_names if name not in mapped_source_names])
        removed_unmapped_deform_count = 0

    return {
        "created_count": created_count,
        "mapped_source_count": len(mapped_source_names),
        "merged_count": sum(1 for source_names in mapping_by_target.values() if len(source_names) > 1),
        "removed_non_deform_count": removed_non_deform_count,
        "removed_unmapped_deform_count": removed_unmapped_deform_count,
        "skipped_conflict_count": skipped_conflict_count,
        "skipped_missing_count": skipped_missing_count,
    }
