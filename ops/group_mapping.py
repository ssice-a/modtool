"""Operator for spatial vertex-group rename mapping."""

import bpy

from ..core.common import ensure_object_mode, restore_context_mode
from ..core.group_mapping import (
    analyze_group_mapping,
    apply_group_mapping_rename,
    count_group_match_statuses,
    resolve_group_mapping_objects,
    store_group_match_rows,
)
from ..core.seam import unify_seam_vertex_weights


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

        status_counts = count_group_match_statuses(scene)
        removed_count = result["removed_non_deform_count"] + result["removed_unmapped_deform_count"]

        message = (
            f"Renamed to {result['created_count']} target group(s) from "
            f"{result['mapped_source_count']} source group(s). "
            f"Direct: {status_counts['DIRECT']}, "
            f"Merge: {status_counts['MERGE']}, "
            f"Fallback: {status_counts['FALLBACK']}."
        )
        if removed_count > 0:
            message += f" Removed {removed_count} helper/unmapped group(s)."
        if result["skipped_conflict_count"] > 0:
            message += f" Skipped {result['skipped_conflict_count']} conflicting row(s)."
        if result["skipped_missing_count"] > 0:
            message += f" Skipped {result['skipped_missing_count']} missing source row(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_unify_seam_weights_from_target(bpy.types.Operator):
    """Copy matched target seam vertex weights onto source seam vertices."""

    bl_idname = "mdt.unify_seam_weights_from_target"
    bl_label = "Unify Seam Weights"
    bl_description = "Copy target weights to matching source seam vertices using nearby open-boundary vertices"
    bl_options = {"REGISTER", "UNDO"}

    match_tolerance: bpy.props.FloatProperty(
        name="Match Tolerance",
        description="Maximum world-space distance for matching source and target seam vertices",
        default=0.0001,
        min=0.0,
        soft_min=0.000001,
        soft_max=0.01,
        precision=6,
    )

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        source_obj, target_obj = resolve_group_mapping_objects(context.scene.mdt_group_match_settings)
        if source_obj is None or target_obj is None:
            self.report({"ERROR"}, "Choose valid Source and Target mesh objects.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = unify_seam_vertex_weights(
                source_obj,
                target_obj,
                match_tolerance=self.match_tolerance,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        if result["written_weight_count"] == 0:
            self.report({"WARNING"}, "Matched seam vertices had no target weights to copy.")
            return {"CANCELLED"}

        message = (
            f"Unified {result['touched_source_vertex_count']} source seam vertex/vertices from "
            f"{target_obj.name}, wrote {result['written_weight_count']} weight(s) across "
            f"{result['copied_group_count']} group(s)."
        )
        if result["unmatched_source_count"] > 0:
            message += f" Unmatched source vertices: {result['unmatched_source_count']}."
        self.report({"INFO"}, message)
        return {"FINISHED"}
