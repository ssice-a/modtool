"""Rename planning helpers for the vertex-group workflow."""

from __future__ import annotations

import re

from .common import sync_edit_mesh_data


_WEIGHT_EPSILON = 1.0e-8


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
