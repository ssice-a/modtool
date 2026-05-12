"""Rename operators for Mod Tool."""

import bpy

from ..core.common import is_mesh_object
from ..core.rename import (
    apply_rename_plan,
    build_rename_plan,
    has_rename_rule,
    remove_empty_vertex_groups,
    remove_non_deforming_vertex_groups,
    reset_rename_settings,
    resolve_final_names,
)


class MDT_OT_batch_rename_groups(bpy.types.Operator):
    """Batch rename the active mesh object's vertex groups."""

    bl_idname = "mdt.batch_rename_groups"
    bl_label = "Apply Rename"
    bl_description = "Rename vertex groups on the active mesh object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return is_mesh_object(obj) and len(getattr(obj, "vertex_groups", [])) > 0

    def execute(self, context):
        obj = context.object
        settings = context.scene.mdt_rename_settings

        if settings.mode == "MODIFY" and not has_rename_rule(settings):
            self.report({"WARNING"}, "Set Search, Prefix, or Suffix first.")
            return {"CANCELLED"}

        try:
            rename_pairs = build_rename_plan(list(obj.vertex_groups), settings)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        resolved_pairs, adjusted_count = resolve_final_names(rename_pairs)
        changed_count = sum(1 for group, final_name in resolved_pairs if group.name != final_name)
        if changed_count == 0:
            self.report({"INFO"}, "No vertex-group names need to change.")
            return {"CANCELLED"}

        renamed_count = apply_rename_plan(resolved_pairs)
        message = f"Renamed {renamed_count} vertex group(s)."
        if adjusted_count:
            message += f" Added automatic suffixes to {adjusted_count} name(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_reset_rename_settings(bpy.types.Operator):
    """Reset rename settings to defaults."""

    bl_idname = "mdt.reset_rename_settings"
    bl_label = "Reset"
    bl_description = "Restore the default rename settings"

    def execute(self, context):
        reset_rename_settings(context.scene.mdt_rename_settings)
        self.report({"INFO"}, "Rename settings were reset.")
        return {"FINISHED"}


class MDT_OT_remove_empty_vertex_groups(bpy.types.Operator):
    """Remove empty vertex groups from the active mesh object."""

    bl_idname = "mdt.remove_empty_vertex_groups"
    bl_label = "Remove Empty"
    bl_description = "Delete vertex groups that have no effective weights on any vertex"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return is_mesh_object(obj) and len(getattr(obj, "vertex_groups", [])) > 0

    def execute(self, context):
        obj = context.object

        removed_count = remove_empty_vertex_groups(obj)
        if removed_count == 0:
            self.report({"INFO"}, "No empty vertex groups were found.")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Removed {removed_count} empty vertex group(s).")
        return {"FINISHED"}


class MDT_OT_keep_deforming_vertex_groups(bpy.types.Operator):
    """Keep only weighted vertex groups that are controlled by deform bones."""

    bl_idname = "mdt.keep_deforming_vertex_groups"
    bl_label = "Keep Bone Groups"
    bl_description = "Delete helper and empty groups, keeping only weighted groups that match deform bones on Armature modifiers"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return is_mesh_object(obj) and len(getattr(obj, "vertex_groups", [])) > 0

    def execute(self, context):
        obj = context.object
        try:
            result = remove_non_deforming_vertex_groups(obj)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if result["removed_count"] == 0:
            self.report({"INFO"}, "No helper or empty deform vertex groups were found.")
            return {"CANCELLED"}

        message = (
            f"Kept {result['kept_count']} weighted deform group(s), "
            f"removed {result['removed_count']} other group(s)."
        )
        if result["removed_empty_deform_count"] > 0:
            message += f" Empty deform groups: {result['removed_empty_deform_count']}."
        self.report({"INFO"}, message)
        return {"FINISHED"}
