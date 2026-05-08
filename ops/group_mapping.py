"""Operator for spatial vertex-group rename mapping."""

import bpy

from ..core.common import ensure_object_mode, restore_context_mode
from ..core.group_mapping import (
    analyze_group_mapping,
    apply_group_mapping_rename,
    count_group_match_status,
    resolve_group_mapping_objects,
    store_group_match_rows,
)


class MDT_OT_rename_groups_from_target(bpy.types.Operator):
    """Analyze and apply source-to-target vertex-group names in one step."""

    bl_idname = "mdt.rename_groups_from_target"
    bl_label = "Rename From Target"
    bl_description = "Rename and merge source vertex groups to target names using spatial weight-cloud matching"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        scene = context.scene
        settings = scene.mdt_group_match_settings
        source_obj, target_obj = resolve_group_mapping_objects(settings)

        if source_obj is None or target_obj is None:
            self.report({"ERROR"}, "Choose valid Source and Target mesh objects.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            analysis = analyze_group_mapping(
                source_obj,
                target_obj,
                depsgraph=context.evaluated_depsgraph_get(),
            )
            store_group_match_rows(scene, analysis["rows"])
            result = apply_group_mapping_rename(source_obj, target_obj, scene.mdt_group_match_results)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        if result["created_count"] == 0:
            self.report({"WARNING"}, "No target vertex-group names could be applied.")
            return {"CANCELLED"}

        direct_count = count_group_match_status(scene, "DIRECT")
        merge_count = count_group_match_status(scene, "MERGE")
        fallback_count = count_group_match_status(scene, "FALLBACK")
        removed_count = result["removed_non_deform_count"] + result["removed_unmapped_deform_count"]

        message = (
            f"Renamed to {result['created_count']} target group(s) from "
            f"{result['mapped_source_count']} source group(s). "
            f"Direct: {direct_count}, Merge: {merge_count}, Fallback: {fallback_count}."
        )
        if removed_count > 0:
            message += f" Removed {removed_count} helper/unmapped group(s)."
        if result["skipped_conflict_count"] > 0:
            message += f" Skipped {result['skipped_conflict_count']} conflicting row(s)."
        if result["skipped_missing_count"] > 0:
            message += f" Skipped {result['skipped_missing_count']} missing source row(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}
