"""Operators for applying modifiers while preserving shape keys."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.modifier_apply import (
    apply_modifier_with_shape_keys,
    get_enabled_modifier_names,
    has_shape_keys,
    normalize_shape_key_ranges,
    set_current_shape_key_values_as_zero,
)


def _summarize_names(names, limit=3):
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} (+{len(names) - limit})"


def _selected_shape_key_objects(context):
    selected_mesh_objects = get_selected_mesh_objects(context)
    shape_key_objects = [obj for obj in selected_mesh_objects if has_shape_keys(obj)]
    if shape_key_objects:
        return shape_key_objects, [
            obj.name
            for obj in selected_mesh_objects
            if not has_shape_keys(obj)
        ]

    obj = context.object
    if has_shape_keys(obj):
        return [obj], []
    return [], [obj.name] if obj is not None else []


def _execute_shape_key_range_operation(operator, context, operation, action_label):
    target_objects, skipped_no_shape_keys = _selected_shape_key_objects(context)
    if len(target_objects) == 0:
        operator.report({"ERROR"}, "Select at least one mesh object with shape keys.")
        return {"CANCELLED"}

    try:
        previous_mode = ensure_object_mode(context)
    except RuntimeError as exc:
        operator.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
        return {"CANCELLED"}

    results = []
    failures = []
    try:
        for obj in target_objects:
            try:
                results.append((obj.name, operation(context, obj)))
            except ValueError as exc:
                failures.append(f"{obj.name}: {exc}")
    finally:
        restore_context_mode(previous_mode)

    if len(results) == 0:
        operator.report({"ERROR"}, failures[0] if failures else "No selected object could be processed.")
        return {"CANCELLED"}

    if skipped_no_shape_keys:
        operator.report(
            {"WARNING"},
            f"Skipped objects without shape keys: {_summarize_names(skipped_no_shape_keys)}.",
        )
    if failures:
        operator.report({"WARNING"}, f"Failed objects: {_summarize_names(failures)}.")

    shape_key_count = sum(result["shape_key_count"] for _obj_name, result in results)
    operator.report(
        {"INFO"},
        f"{action_label} {shape_key_count} shape key(s) on {len(results)} object(s).",
    )
    return {"FINISHED"}


class MDT_OT_apply_modifier_keep_shape_keys(bpy.types.Operator):
    """Apply enabled modifiers on meshes with shape keys without deleting them."""

    bl_idname = "mdt.apply_modifier_keep_shape_keys"
    bl_label = "Apply Enabled Modifiers"
    bl_description = (
        "Apply every viewport-enabled modifier, including data-transfer and vertex-group changes, while rebuilding shape keys"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        selected_mesh_objects = get_selected_mesh_objects(context)
        eligible_objects = []
        skipped_no_shape_keys = []
        skipped_no_modifiers = []

        for obj in selected_mesh_objects:
            if not has_shape_keys(obj):
                skipped_no_shape_keys.append(obj.name)
                continue
            if len(get_enabled_modifier_names(obj)) == 0:
                skipped_no_modifiers.append(obj.name)
                continue
            eligible_objects.append(obj)

        if len(eligible_objects) == 0:
            self.report(
                {"ERROR"},
                "Select at least one mesh object that has shape keys and viewport-enabled modifiers.",
            )
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        results = []
        failures = []
        try:
            for obj in eligible_objects:
                try:
                    results.append((obj.name, apply_modifier_with_shape_keys(context, obj)))
                except ValueError as exc:
                    failures.append(f"{obj.name}: {exc}")
        finally:
            restore_context_mode(previous_mode)

        if len(results) == 0:
            self.report({"ERROR"}, failures[0] if failures else "No selected object could be processed.")
            return {"CANCELLED"}

        if skipped_no_shape_keys:
            self.report(
                {"WARNING"},
                f"Skipped objects without shape keys: {_summarize_names(skipped_no_shape_keys)}.",
            )
        if skipped_no_modifiers:
            self.report(
                {"WARNING"},
                f"Skipped objects without viewport-enabled modifiers: {_summarize_names(skipped_no_modifiers)}.",
            )
        if failures:
            self.report(
                {"WARNING"},
                f"Failed objects: {_summarize_names(failures)}.",
            )

        total_modifier_count = sum(len(result["applied_modifier_names"]) for _obj_name, result in results)
        total_shape_key_count = sum(result["shape_key_count"] for _obj_name, result in results)
        total_vertex_group_count = sum(result["vertex_group_count"] for _obj_name, result in results)
        action_count = sum(1 for _obj_name, result in results if result["action_copied"])
        driver_count = sum(result["driver_count"] for _obj_name, result in results)

        message = (
            f"Applied {total_modifier_count} modifier(s) on {len(results)} selected object(s). "
            f"Preserved {total_shape_key_count} shape key(s)."
        )
        if total_vertex_group_count > 0:
            message += f" Baked evaluated data for {total_vertex_group_count} vertex group(s)."
        if action_count > 0:
            message += f" Reattached actions on {action_count} object(s)."
        if driver_count > 0:
            message += f" Rebuilt {driver_count} driver(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_set_current_shape_key_values_as_zero(bpy.types.Operator):
    """Make the current shape-key mix the new zero point."""

    bl_idname = "mdt.set_shape_key_current_values_as_zero"
    bl_label = "Current Values as Zero"
    bl_description = "Move the current relative shape-key values into Basis, then shift every slider around the new zero"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        return _execute_shape_key_range_operation(
            self,
            context,
            set_current_shape_key_values_as_zero,
            "Set current values as zero for",
        )


class MDT_OT_normalize_shape_key_ranges(bpy.types.Operator):
    """Normalize every shape-key slider range to 0..1."""

    bl_idname = "mdt.normalize_shape_key_ranges"
    bl_label = "Normalize Ranges 0-1"
    bl_description = "Remap each relative shape-key slider to 0..1 without changing the visible mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        return _execute_shape_key_range_operation(
            self,
            context,
            normalize_shape_key_ranges,
            "Normalized ranges for",
        )
