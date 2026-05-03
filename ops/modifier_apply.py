"""Operators for applying modifiers while preserving shape keys."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.modifier_apply import (
    apply_modifier_with_shape_keys,
    delete_selected_geometry_with_shape_keys,
    get_enabled_modifier_names,
    has_shape_keys,
)


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

    @staticmethod
    def _summarize_names(names, limit=3):
        if len(names) <= limit:
            return ", ".join(names)
        return f"{', '.join(names[:limit])} (+{len(names) - limit})"

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
                f"Skipped objects without shape keys: {self._summarize_names(skipped_no_shape_keys)}.",
            )
        if skipped_no_modifiers:
            self.report(
                {"WARNING"},
                f"Skipped objects without viewport-enabled modifiers: {self._summarize_names(skipped_no_modifiers)}.",
            )
        if failures:
            self.report(
                {"WARNING"},
                f"Failed objects: {self._summarize_names(failures)}.",
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


class MDT_OT_delete_selected_geometry_keep_shape_keys(bpy.types.Operator):
    """Delete selected mesh geometry while preserving existing shape keys."""

    bl_idname = "mdt.delete_selected_geometry_keep_shape_keys"
    bl_label = "Delete Selection"
    bl_description = "Delete selected vertices, edges, or faces while rebuilding shape keys for the remaining mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return has_shape_keys(context.object)

    def execute(self, context):
        obj = context.object

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = delete_selected_geometry_with_shape_keys(context, obj)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        message = (
            f"Deleted {result['deleted_vertex_count']} vertex/vertices. "
            f"Preserved {result['shape_key_count']} shape key(s)."
        )
        if result["driver_count"] > 0:
            message += f" Rebuilt {result['driver_count']} driver(s)."
        if result["action_copied"]:
            message += " Reattached shape-key action."
        self.report({"INFO"}, message)
        return {"FINISHED"}
