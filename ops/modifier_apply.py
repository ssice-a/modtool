"""Operators for applying modifiers while preserving shape keys."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.modifier_apply import (
    apply_unlocked_shape_keys,
    apply_enabled_modifiers,
    get_enabled_modifier_names,
    has_shape_keys,
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


class MDT_OT_apply_modifier_keep_shape_keys(bpy.types.Operator):
    """Apply enabled modifiers on every selected mesh, preserving any shape keys."""

    bl_idname = "mdt.apply_modifier_keep_shape_keys"
    bl_label = "Apply Enabled Modifiers"
    bl_description = (
        "Apply viewport-enabled modifiers on all selected meshes, preserving existing shape keys"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        selected_mesh_objects = get_selected_mesh_objects(context)
        eligible_objects = []
        skipped_no_modifiers = []

        for obj in selected_mesh_objects:
            if len(get_enabled_modifier_names(obj)) == 0:
                skipped_no_modifiers.append(obj.name)
                continue
            eligible_objects.append(obj)

        if len(eligible_objects) == 0:
            self.report(
                {"ERROR"},
                "Select at least one mesh object with viewport-enabled modifiers.",
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
                    results.append((obj.name, apply_enabled_modifiers(context, obj)))
                except (ValueError, RuntimeError) as exc:
                    failures.append(f"{obj.name}: {exc}")
        finally:
            restore_context_mode(previous_mode)

        if len(results) == 0:
            self.report({"ERROR"}, failures[0] if failures else "No selected object could be processed.")
            return {"CANCELLED"}

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


class MDT_OT_apply_unlocked_shape_keys(bpy.types.Operator):
    """Bake current unlocked shape-key contributions into Basis, then remove those keys."""

    bl_idname = "mdt.apply_unlocked_shape_keys"
    bl_label = "Apply Unlocked Shape Keys"
    bl_description = "Bake current unlocked shape-key values into Basis and remove those shape keys"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        target_objects, skipped_no_shape_keys = _selected_shape_key_objects(context)
        if len(target_objects) == 0:
            self.report({"ERROR"}, "Select at least one mesh object with shape keys.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        results = []
        failures = []
        try:
            for obj in target_objects:
                try:
                    results.append((obj.name, apply_unlocked_shape_keys(context, obj)))
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
        if failures:
            self.report({"WARNING"}, f"Failed objects: {_summarize_names(failures)}.")

        applied_shape_key_count = sum(result["applied_shape_key_count"] for _obj_name, result in results)
        applied_value_count = sum(result["applied_value_count"] for _obj_name, result in results)
        self.report(
            {"INFO"},
            f"Applied {applied_shape_key_count} unlocked shape key(s) on {len(results)} object(s). "
            f"Baked {applied_value_count} current contribution(s).",
        )
        return {"FINISHED"}
