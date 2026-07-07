"""Rename planning helpers for the vertex-group workflow."""

from __future__ import annotations

from array import array
import re

from mathutils import Vector
from mathutils.kdtree import KDTree

from .common import get_flat_point_coordinates, is_armature_object, sync_edit_mesh_data


_WEIGHT_EPSILON = 1.0e-8

_BIP001_BODY_EXACT_PARTS = {
    "pelvis",
    "spine",
    "spine1",
    "spine2",
    "neck",
    "head",
    "l-clavicle",
    "r-clavicle",
    "l-upperarm",
    "r-upperarm",
    "l-forearm",
    "r-forearm",
    "l-hand",
    "r-hand",
    "l-thigh",
    "r-thigh",
    "l-calf",
    "r-calf",
    "l-foot",
    "r-foot",
}
_BIP001_BODY_PREFIX_PARTS = (
    "l-finger",
    "r-finger",
    "l-toe",
    "r-toe",
)
_BODY_TWIST_PART_TOKENS = (
    "upperarm",
    "forearm",
    "thigh",
    "calf",
)
_BODY_BN_TOKENS = (
    "handd",
    "handu",
    "elbowavg",
    "elbowpush",
    "kneeavg",
    "kneepush",
    "xiong",
)
_NON_BODY_GROUP_TOKENS = (
    "cloth",
    "qun",
    "skirt",
    "dress",
    "tail",
    "piaodai",
    "lalian",
    "gongpai",
    "tie",
    "callor",
    "collar",
    "hair",
    "weapon",
    "prop",
    "socket",
    "fx",
)
_SIDE_NAME_REPLACEMENTS = (
    ("-L-", "-R-"),
    ("-R-", "-L-"),
    ("-l-", "-r-"),
    ("-r-", "-l-"),
    ("_L_", "_R_"),
    ("_R_", "_L_"),
    ("_l_", "_r_"),
    ("_r_", "_l_"),
)
_SIDE_NAME_SUFFIX_REPLACEMENTS = (
    ("_L", "_R"),
    ("_R", "_L"),
    ("_l", "_r"),
    ("_r", "_l"),
    (".L", ".R"),
    (".R", ".L"),
    (".l", ".r"),
    (".r", ".l"),
    ("-L", "-R"),
    ("-R", "-L"),
    ("-l", "-r"),
    ("-r", "-l"),
)
_LEFT_SIDE_TOKENS = ("-L-", "-l-", "_L_", "_l_")
_RIGHT_SIDE_TOKENS = ("-R-", "-r-", "_R_", "_r_")
_LEFT_SIDE_SUFFIXES = ("_L", "_l", ".L", ".l", "-L", "-l")
_RIGHT_SIDE_SUFFIXES = ("_R", "_r", ".R", ".r", "-R", "-r")


def has_rename_rule(settings):
    """Return True when rule mode has at least one effective rename step."""
    return bool(settings.search or settings.prefix or settings.suffix)


def reset_rename_settings(settings):
    """Restore rename settings to their defaults."""
    settings.mode = "MODIFY"
    settings.search = ""
    settings.replace = ""
    settings.use_regex = False
    settings.case_sensitive = False
    settings.prefix = ""
    settings.suffix = ""
    settings.sequence_base = "Group"
    settings.sequence_separator = "_"
    settings.sequence_start = 1
    settings.sequence_padding = 2


def _compile_regex(settings):
    flags = 0 if settings.case_sensitive else re.IGNORECASE
    return re.compile(settings.search, flags)


def _apply_modify_rule(name, settings):
    updated_name = name
    if settings.search:
        if settings.use_regex:
            updated_name = _compile_regex(settings).sub(settings.replace, updated_name)
        elif settings.case_sensitive:
            updated_name = updated_name.replace(settings.search, settings.replace)
        else:
            updated_name = re.sub(
                re.escape(settings.search),
                lambda _match: settings.replace,
                updated_name,
                flags=re.IGNORECASE,
            )
    return f"{settings.prefix}{updated_name}{settings.suffix}"


def _build_sequence_name(index, settings):
    sequence_number = str(settings.sequence_start + index).zfill(settings.sequence_padding)
    if settings.sequence_base:
        if settings.sequence_separator:
            return f"{settings.sequence_base}{settings.sequence_separator}{sequence_number}"
        return f"{settings.sequence_base}{sequence_number}"
    return sequence_number


def build_rename_plan(vertex_groups, settings):
    """Build a desired rename plan for the provided vertex groups."""
    rename_pairs = []
    if settings.mode == "SEQUENCE":
        for index, group in enumerate(vertex_groups):
            rename_pairs.append((group, _build_sequence_name(index, settings)))
        return rename_pairs

    if settings.use_regex and settings.search:
        try:
            _compile_regex(settings)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc

    for group in vertex_groups:
        rename_pairs.append((group, _apply_modify_rule(group.name, settings)))
    return rename_pairs


def _make_unique_name(base_name, used_names):
    if base_name not in used_names:
        return base_name

    suffix_index = 1
    while True:
        candidate_name = f"{base_name}.{suffix_index:03d}"
        if candidate_name not in used_names:
            return candidate_name
        suffix_index += 1


def resolve_final_names(rename_pairs):
    """Resolve collisions against unchanged groups and other renamed groups."""
    used_names = {target_name for group, target_name in rename_pairs if group.name == target_name}
    resolved_pairs = []
    adjusted_count = 0

    for group, target_name in rename_pairs:
        if group.name == target_name:
            resolved_pairs.append((group, target_name))
            continue

        final_name = _make_unique_name(target_name, used_names)
        if final_name != target_name:
            adjusted_count += 1
        used_names.add(final_name)
        resolved_pairs.append((group, final_name))

    return resolved_pairs, adjusted_count


def _build_temp_name(used_names, index):
    temp_name = f"__MDT_TMP_{index}__"
    if temp_name not in used_names:
        return temp_name

    suffix_index = 1
    while True:
        candidate_name = f"__MDT_TMP_{index}_{suffix_index}__"
        if candidate_name not in used_names:
            return candidate_name
        suffix_index += 1


def apply_rename_plan(resolved_pairs):
    """Apply renaming in two phases to avoid intermediate name collisions."""
    changed_pairs = [(group, final_name) for group, final_name in resolved_pairs if group.name != final_name]
    if not changed_pairs:
        return 0

    used_names = {group.name for group, _final_name in resolved_pairs}
    used_names.update(final_name for _group, final_name in resolved_pairs)
    temporary_pairs = []
    for index, (group, _final_name) in enumerate(changed_pairs):
        temp_name = _build_temp_name(used_names, index)
        used_names.add(temp_name)
        temporary_pairs.append((group, temp_name))

    for group, temp_name in temporary_pairs:
        group.name = temp_name

    for group, final_name in changed_pairs:
        group.name = final_name

    return len(changed_pairs)


def apply_exact_name_pairs(obj, name_pairs):
    """Rename vertex groups to exact target names while skipping unsafe conflicts."""
    name_pairs = list(name_pairs)
    if obj is None:
        return {
            "renamed_count": 0,
            "skipped_conflict_count": 0,
            "skipped_missing_count": len(name_pairs),
        }

    source_groups_by_name = {group.name: group for group in obj.vertex_groups}
    raw_pairs = []
    skipped_missing_count = 0

    for source_name, target_name in name_pairs:
        if not source_name or not target_name:
            skipped_missing_count += 1
            continue

        source_group = source_groups_by_name.get(source_name)
        if source_group is None:
            skipped_missing_count += 1
            continue
        if source_group.name == target_name:
            continue
        raw_pairs.append((source_group, target_name))

    if not raw_pairs:
        return {
            "renamed_count": 0,
            "skipped_conflict_count": 0,
            "skipped_missing_count": skipped_missing_count,
        }

    rename_source_names = {group.name for group, _target_name in raw_pairs}
    existing_names = {group.name for group in obj.vertex_groups}
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

    return {
        "renamed_count": apply_rename_plan(resolved_pairs),
        "skipped_conflict_count": skipped_conflict_count,
        "skipped_missing_count": skipped_missing_count,
    }


def _collect_weighted_group_indices(obj):
    used_group_indices = set()
    for vertex in getattr(obj.data, "vertices", []):
        for assignment in vertex.groups:
            if assignment.weight > _WEIGHT_EPSILON:
                used_group_indices.add(assignment.group)
    return used_group_indices


def remove_empty_vertex_groups(obj):
    """Remove vertex groups that have no effective non-zero weights on any vertex."""
    if obj is None:
        return 0

    sync_edit_mesh_data(obj)
    used_group_indices = _collect_weighted_group_indices(obj)

    removed_count = 0
    for group in reversed(list(obj.vertex_groups)):
        if group.index in used_group_indices:
            continue
        obj.vertex_groups.remove(group)
        removed_count += 1

    return removed_count


def _build_mesh_adjacency(mesh):
    adjacency = [[] for _vertex in mesh.vertices]
    if not mesh.edges:
        return adjacency

    edge_vertices = array("i", [0]) * (len(mesh.edges) * 2)
    mesh.edges.foreach_get("vertices", edge_vertices)
    for edge_index in range(0, len(edge_vertices), 2):
        vertex_a = edge_vertices[edge_index]
        vertex_b = edge_vertices[edge_index + 1]
        if vertex_a == vertex_b:
            continue
        adjacency[vertex_a].append(vertex_b)
        adjacency[vertex_b].append(vertex_a)
    return adjacency


def _collect_vertex_group_weight_entries(obj):
    entries_by_group_index = [[] for _group in obj.vertex_groups]
    group_count = len(entries_by_group_index)
    for vertex in obj.data.vertices:
        vertex_index = vertex.index
        for assignment in vertex.groups:
            if assignment.weight <= _WEIGHT_EPSILON:
                continue
            if assignment.group < 0 or assignment.group >= group_count:
                continue
            entries_by_group_index[assignment.group].append((vertex_index, float(assignment.weight)))
    return entries_by_group_index


def _weighted_components(weight_by_vertex_index, adjacency):
    unvisited = set(weight_by_vertex_index)
    components = []

    while unvisited:
        start_vertex = unvisited.pop()
        stack = [start_vertex]
        component_vertices = [start_vertex]
        component_weight = weight_by_vertex_index[start_vertex]

        while stack:
            vertex_index = stack.pop()
            for neighbor_index in adjacency[vertex_index]:
                if neighbor_index not in unvisited:
                    continue
                unvisited.remove(neighbor_index)
                stack.append(neighbor_index)
                component_vertices.append(neighbor_index)
                component_weight += weight_by_vertex_index[neighbor_index]

        components.append((component_weight, len(component_vertices), component_vertices))

    return components


def clean_disconnected_vertex_group_weights(obj):
    """Keep only the largest connected weight island in each vertex group."""
    if obj is None:
        raise ValueError("Choose a valid mesh object.")
    if not getattr(obj, "vertex_groups", None):
        raise ValueError("The active mesh has no vertex groups.")

    sync_edit_mesh_data(obj)
    adjacency = _build_mesh_adjacency(obj.data)
    entries_by_group_index = _collect_vertex_group_weight_entries(obj)

    processed_group_count = 0
    cleaned_group_count = 0
    removed_weight_count = 0
    removed_component_count = 0

    for group in obj.vertex_groups:
        entries = entries_by_group_index[group.index]
        if len(entries) <= 1:
            continue

        processed_group_count += 1
        weight_by_vertex_index = dict(entries)
        components = _weighted_components(weight_by_vertex_index, adjacency)
        if len(components) <= 1:
            continue

        keep_index = max(
            range(len(components)),
            key=lambda component_index: (
                components[component_index][0],
                components[component_index][1],
            ),
        )
        remove_vertex_indices = []
        for component_index, _component in enumerate(components):
            if component_index == keep_index:
                continue
            remove_vertex_indices.extend(components[component_index][2])

        if not remove_vertex_indices:
            continue

        group.remove(remove_vertex_indices)
        cleaned_group_count += 1
        removed_weight_count += len(remove_vertex_indices)
        removed_component_count += len(components) - 1

    return {
        "processed_group_count": processed_group_count,
        "cleaned_group_count": cleaned_group_count,
        "removed_weight_count": removed_weight_count,
        "removed_component_count": removed_component_count,
    }


def _resolve_deforming_bone_names(obj):
    bone_names = set()
    for modifier in getattr(obj, "modifiers", []):
        if getattr(modifier, "type", None) != "ARMATURE":
            continue
        if not getattr(modifier, "use_vertex_groups", True):
            continue
        armature_obj = getattr(modifier, "object", None)
        if getattr(armature_obj, "type", None) != "ARMATURE":
            continue
        for bone in getattr(armature_obj.data, "bones", []):
            if getattr(bone, "use_deform", False):
                bone_names.add(bone.name)
    return bone_names


def _keepable_deforming_group_names(obj):
    deforming_bone_names = _resolve_deforming_bone_names(obj)
    if not deforming_bone_names:
        return set(), set()

    weighted_group_indices = _collect_weighted_group_indices(obj)
    keep_names = {
        group.name
        for group in getattr(obj, "vertex_groups", [])
        if group.name in deforming_bone_names and group.index in weighted_group_indices
    }
    return keep_names, deforming_bone_names


def remove_non_deforming_vertex_groups(obj):
    """Keep only weighted vertex groups that correspond to deform bones on Armature modifiers."""
    if obj is None:
        raise ValueError("Choose a valid mesh object.")

    sync_edit_mesh_data(obj)
    keep_names, deforming_bone_names = _keepable_deforming_group_names(obj)
    if not deforming_bone_names:
        raise ValueError("The active mesh needs an Armature modifier with deform bones.")

    removed_count = 0
    removed_helper_count = 0
    removed_empty_deform_count = 0
    for group in reversed(list(obj.vertex_groups)):
        if group.name in keep_names:
            continue
        if group.name in deforming_bone_names:
            removed_empty_deform_count += 1
        else:
            removed_helper_count += 1
        obj.vertex_groups.remove(group)
        removed_count += 1

    return {
        "kept_count": len(keep_names),
        "removed_count": removed_count,
        "removed_helper_count": removed_helper_count,
        "removed_empty_deform_count": removed_empty_deform_count,
    }


def _is_bip001_body_group(normalized_name):
    if not normalized_name.startswith("bip001-"):
        return False

    part_name = normalized_name[len("bip001-") :]
    return part_name in _BIP001_BODY_EXACT_PARTS or part_name.startswith(_BIP001_BODY_PREFIX_PARTS)


def _is_body_twist_group(normalized_name):
    if not normalized_name.startswith("bone-") or "twist" not in normalized_name:
        return False
    return any(token in normalized_name for token in _BODY_TWIST_PART_TOKENS)


def _is_body_bn_group(normalized_name):
    if not normalized_name.startswith("bn_"):
        return False
    compact_name = normalized_name.replace("_", "")
    return any(token in compact_name for token in _BODY_BN_TOKENS)


def is_body_vertex_group_name(name):
    """Return True when a game-style vertex group belongs to the body skeleton."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        return False
    if any(token in normalized_name for token in _NON_BODY_GROUP_TOKENS):
        return False
    return (
        _is_bip001_body_group(normalized_name)
        or _is_body_twist_group(normalized_name)
        or _is_body_bn_group(normalized_name)
    )


def _find_best_group_armature(group_names, armature_objects):
    best_armature = None
    best_match_count = 0

    for armature_obj in armature_objects or ():
        if not is_armature_object(armature_obj):
            continue

        bone_names = {bone.name for bone in getattr(armature_obj.data, "bones", [])}
        match_count = len(group_names & bone_names)
        if match_count > best_match_count:
            best_armature = armature_obj
            best_match_count = match_count

    return best_armature, best_match_count


def _find_body_parent_name(bone, keep_names):
    parent = getattr(bone, "parent", None)
    while parent is not None:
        if parent.name in keep_names:
            return parent.name
        parent = getattr(parent, "parent", None)
    return None


def _build_non_body_parent_map(groups, keep_names, armature_obj):
    if armature_obj is None:
        return {}

    bone_by_name = {bone.name: bone for bone in getattr(armature_obj.data, "bones", [])}
    parent_map = {}
    for group in groups:
        if group.name in keep_names:
            continue

        bone = bone_by_name.get(group.name)
        if bone is None:
            continue

        parent_name = _find_body_parent_name(bone, keep_names)
        if parent_name is not None:
            parent_map[group.name] = parent_name

    return parent_map


def _fold_vertex_group_weights_to_parents(obj, parent_map):
    if not parent_map:
        return {
            "folded_group_count": 0,
            "folded_weight_count": 0,
            "weighted_group_indices": _collect_weighted_group_indices(obj),
        }

    group_by_name = {group.name: group for group in obj.vertex_groups}
    source_index_to_target_group = {}
    for source_name, target_name in parent_map.items():
        source_group = group_by_name.get(source_name)
        target_group = group_by_name.get(target_name)
        if source_group is None or target_group is None:
            continue
        source_index_to_target_group[source_group.index] = target_group

    weighted_group_indices = set()
    folded_weight_count = 0
    for vertex in getattr(obj.data, "vertices", []):
        additions_by_target = {}
        for assignment in list(vertex.groups):
            if assignment.weight <= _WEIGHT_EPSILON:
                continue

            weighted_group_indices.add(assignment.group)
            target_group = source_index_to_target_group.get(assignment.group)
            if target_group is None:
                continue

            additions_by_target[target_group] = additions_by_target.get(target_group, 0.0) + assignment.weight
            folded_weight_count += 1

        for target_group, weight in additions_by_target.items():
            target_group.add((vertex.index,), weight, "ADD")

    return {
        "folded_group_count": len(source_index_to_target_group),
        "folded_weight_count": folded_weight_count,
        "weighted_group_indices": weighted_group_indices,
    }


def remove_non_body_vertex_groups(obj, armature_objects=None):
    """Fold non-body weights to parent body bones, then remove the folded groups."""
    if obj is None:
        raise ValueError("Choose a valid mesh object.")

    sync_edit_mesh_data(obj)

    groups = list(obj.vertex_groups)
    group_names = {group.name for group in groups}
    keep_names = {group.name for group in groups if is_body_vertex_group_name(group.name)}
    if not keep_names:
        raise ValueError("The active mesh has no recognized body vertex groups.")

    armature_obj, armature_match_count = _find_best_group_armature(group_names, armature_objects)
    parent_map = _build_non_body_parent_map(groups, keep_names, armature_obj)
    fold_result = _fold_vertex_group_weights_to_parents(obj, parent_map)
    weighted_group_indices = fold_result["weighted_group_indices"]

    kept_count = 0
    removed_count = 0
    skipped_unmapped_count = 0
    skipped_weighted_unmapped_count = 0
    for group in reversed(list(obj.vertex_groups)):
        if group.name in keep_names:
            kept_count += 1
            continue
        if group.name not in parent_map and group.index in weighted_group_indices:
            skipped_unmapped_count += 1
            skipped_weighted_unmapped_count += 1
            continue
        obj.vertex_groups.remove(group)
        removed_count += 1

    return {
        "kept_count": kept_count,
        "removed_count": removed_count,
        "folded_group_count": fold_result["folded_group_count"],
        "folded_weight_count": fold_result["folded_weight_count"],
        "skipped_unmapped_count": skipped_unmapped_count,
        "skipped_weighted_unmapped_count": skipped_weighted_unmapped_count,
        "armature_name": getattr(armature_obj, "name", ""),
        "armature_match_count": armature_match_count,
    }


def mirror_vertex_group_side_name(name):
    """Return the opposite-side vertex-group name candidate, or None for center names."""
    for source_token, target_token in _SIDE_NAME_REPLACEMENTS:
        if source_token in name:
            return name.replace(source_token, target_token, 1)

    for source_suffix, target_suffix in _SIDE_NAME_SUFFIX_REPLACEMENTS:
        if name.endswith(source_suffix):
            return f"{name[: -len(source_suffix)]}{target_suffix}"

    return None


def _vertex_group_side(name):
    if any(token in name for token in _LEFT_SIDE_TOKENS) or name.endswith(_LEFT_SIDE_SUFFIXES):
        return "LEFT"
    if any(token in name for token in _RIGHT_SIDE_TOKENS) or name.endswith(_RIGHT_SIDE_SUFFIXES):
        return "RIGHT"
    return None


def _build_mirror_group_pairs(obj):
    group_names = {group.name for group in obj.vertex_groups}
    group_pairs = []
    seen_names = set()

    for group in obj.vertex_groups:
        if group.name in seen_names:
            continue

        mirrored_name = mirror_vertex_group_side_name(group.name)
        if mirrored_name is None or mirrored_name not in group_names:
            continue

        seen_names.add(group.name)
        seen_names.add(mirrored_name)
        side = _vertex_group_side(group.name)
        mirrored_side = _vertex_group_side(mirrored_name)
        if side == "LEFT" and mirrored_side == "RIGHT":
            group_pairs.append((group.name, mirrored_name))
        elif side == "RIGHT" and mirrored_side == "LEFT":
            group_pairs.append((mirrored_name, group.name))

    return group_pairs


def _read_sparse_group_weights(obj, group_name):
    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        return {}

    weights_by_vertex_index = {}
    group_index = vertex_group.index
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group_index and assignment.weight > _WEIGHT_EPSILON:
                weights_by_vertex_index[vertex.index] = float(assignment.weight)
                break
    return weights_by_vertex_index


def _clear_group_weights(obj, group_names):
    group_indices = {
        group.index: group
        for group in obj.vertex_groups
        if group.name in group_names
    }
    if not group_indices:
        return 0

    vertices_by_group_index = {group_index: [] for group_index in group_indices}
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group in vertices_by_group_index:
                vertices_by_group_index[assignment.group].append(vertex.index)

    cleared_count = 0
    for group_index, vertex_indices in vertices_by_group_index.items():
        if not vertex_indices:
            continue
        group_indices[group_index].remove(vertex_indices)
        cleared_count += len(vertex_indices)
    return cleared_count


def _write_sparse_group_weights(obj, group_name, weights_by_vertex_index):
    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        return 0

    written_count = 0
    for vertex_index, weight in weights_by_vertex_index.items():
        vertex_group.add((vertex_index,), min(1.0, weight), "REPLACE")
        written_count += 1
    return written_count


def _custom_mirror_tolerance(flat_coordinates):
    if not flat_coordinates:
        return 1.0e-5

    min_x = max_x = flat_coordinates[0]
    min_y = max_y = flat_coordinates[1]
    min_z = max_z = flat_coordinates[2]
    for coordinate_index in range(3, len(flat_coordinates), 3):
        x = flat_coordinates[coordinate_index]
        y = flat_coordinates[coordinate_index + 1]
        z = flat_coordinates[coordinate_index + 2]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    diagonal = Vector((max_x - min_x, max_y - min_y, max_z - min_z)).length
    return max(1.0e-5, diagonal * 0.0005)


def _build_vertex_kdtree(flat_coordinates, vertex_count):
    kd_tree = KDTree(vertex_count)
    for vertex_index in range(vertex_count):
        coordinate_index = vertex_index * 3
        kd_tree.insert(
            Vector(
                (
                    flat_coordinates[coordinate_index],
                    flat_coordinates[coordinate_index + 1],
                    flat_coordinates[coordinate_index + 2],
                )
            ),
            vertex_index,
        )
    kd_tree.balance()
    return kd_tree


def _custom_mirror_weights(obj, source_to_target, tolerance=0.0):
    flat_coordinates = get_flat_point_coordinates(obj.data.vertices)
    vertex_count = len(obj.data.vertices)
    if tolerance <= 0.0:
        tolerance = _custom_mirror_tolerance(flat_coordinates)

    kd_tree = _build_vertex_kdtree(flat_coordinates, vertex_count)
    group_by_name = {group.name: group for group in obj.vertex_groups}
    source_index_to_target_name = {
        group_by_name[source_name].index: target_name
        for source_name, target_name in source_to_target.items()
        if source_name in group_by_name and target_name in group_by_name
    }
    weights_by_target_name = {target_name: {} for target_name in source_to_target.values()}
    mirrored_vertex_by_source = {}
    unmatched_vertex_count = 0

    for vertex in obj.data.vertices:
        source_assignments = [
            assignment
            for assignment in vertex.groups
            if assignment.group in source_index_to_target_name and assignment.weight > _WEIGHT_EPSILON
        ]
        if not source_assignments:
            continue

        coordinate_index = vertex.index * 3
        mirrored_coordinate = Vector(
            (
                -flat_coordinates[coordinate_index],
                flat_coordinates[coordinate_index + 1],
                flat_coordinates[coordinate_index + 2],
            )
        )
        _found_coordinate, target_vertex_index, distance = kd_tree.find(mirrored_coordinate)
        if target_vertex_index is None or distance > tolerance:
            unmatched_vertex_count += 1
            continue

        mirrored_vertex_by_source[vertex.index] = target_vertex_index
        for assignment in source_assignments:
            target_name = source_index_to_target_name[assignment.group]
            target_weights = weights_by_target_name[target_name]
            target_weights[target_vertex_index] = min(
                1.0,
                target_weights.get(target_vertex_index, 0.0) + float(assignment.weight),
            )

    cleared_weight_count = _clear_group_weights(obj, set(weights_by_target_name))
    written_weight_count = 0
    for target_name, weights_by_vertex_index in weights_by_target_name.items():
        written_weight_count += _write_sparse_group_weights(obj, target_name, weights_by_vertex_index)

    return {
        "cleared_weight_count": cleared_weight_count,
        "written_weight_count": written_weight_count,
        "unmatched_vertex_count": unmatched_vertex_count,
        "paired_vertex_count": len(mirrored_vertex_by_source),
        "tolerance": tolerance,
    }


def mirror_vertex_group_weights_x(
    obj,
    direction="LEFT_TO_RIGHT",
    tolerance=0.0,
):
    """Mirror recognized left/right vertex-group pairs without renaming groups."""
    if obj is None:
        raise ValueError("Choose a valid mesh object.")
    if direction not in {"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"}:
        raise ValueError("Choose a valid mirror direction.")

    sync_edit_mesh_data(obj)
    group_pairs = _build_mirror_group_pairs(obj)
    if not group_pairs:
        raise ValueError("No left/right vertex-group pairs were found.")

    source_to_target = {
        left_name if direction == "LEFT_TO_RIGHT" else right_name:
        right_name if direction == "LEFT_TO_RIGHT" else left_name
        for left_name, right_name in group_pairs
    }

    result = _custom_mirror_weights(obj, source_to_target, tolerance=tolerance)
    return {
        "pair_count": len(group_pairs),
        "mirrored_group_count": len(source_to_target),
        **result,
    }


def get_rename_preview(obj, settings, limit=5):
    """Return preview rows, summary counts, and validation errors for the UI."""
    vertex_groups = list(getattr(obj, "vertex_groups", []))
    if not vertex_groups:
        return {
            "rows": [],
            "changed_count": 0,
            "total_count": 0,
            "adjusted_count": 0,
            "error": "",
        }

    try:
        rename_pairs = build_rename_plan(vertex_groups, settings)
    except ValueError as exc:
        return {
            "rows": [],
            "changed_count": 0,
            "total_count": len(vertex_groups),
            "adjusted_count": 0,
            "error": str(exc),
        }

    resolved_pairs, adjusted_count = resolve_final_names(rename_pairs)
    changed_count = sum(1 for group, final_name in resolved_pairs if group.name != final_name)
    preview_rows = [
        (group.name, final_name, True)
        for group, final_name in resolved_pairs
        if group.name != final_name
    ][:limit]

    return {
        "rows": preview_rows,
        "changed_count": changed_count,
        "total_count": len(vertex_groups),
        "adjusted_count": adjusted_count,
        "error": "",
    }
