"""Operators for analyzing and applying vertex-group rename mappings."""

import bpy

from ..core.common import ensure_object_mode, restore_context_mode
from ..core.group_mapping import (
    analyze_group_mapping,
    apply_group_mapping_rename,
    count_applicable_group_matches,
    count_group_match_status,
    resolve_group_mapping_objects,
    store_group_match_rows,
)


class MDT_OT_analyze_group_mapping(bpy.types.Operator):
    """Analyze source-to-target vertex-group rename suggestions."""

    bl_idname = "mdt.analyze_group_mapping"
    bl_label = "Analyze Mapping"
    bl_description = "Analyze source and target meshes and build a safe vertex-group rename table"
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
                candidate_limit=settings.candidate_limit,
                surface_tolerance=settings.surface_tolerance,
                prefer_deform_groups=settings.prefer_deform_groups,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        store_group_match_rows(scene, analysis["rows"])
        direct_count = count_group_match_status(scene, "DIRECT")
        ambiguous_count = count_group_match_status(scene, "AMBIGUOUS")
        unmatched_count = count_group_match_status(scene, "UNMATCHED")
        conflict_count = count_group_match_status(scene, "CONFLICT")

        message = (
            f"Analyzed {analysis['source_group_count']} source group(s). "
            f"Direct: {direct_count}, Ambiguous: {ambiguous_count}, "
            f"Unmatched: {unmatched_count}, Conflict: {conflict_count}."
        )
        if analysis["ignored_triangle_count"] > 0:
            message += f" Ignored {analysis['ignored_triangle_count']} degenerate target triangle(s)."
        if analysis["unmatched_vertex_count"] > 0:
            message += (
                f" Unmatched source vertices: {analysis['unmatched_vertex_count']} "
                f"(farthest hit: {analysis['farthest_distance']:.6f})."
            )
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_apply_group_mapping_rename(bpy.types.Operator):
    """Apply high-confidence rename suggestions from the stored mapping table."""

    bl_idname = "mdt.apply_group_mapping_rename"
    bl_label = "Apply Mapping Rename"
    bl_description = "Rename source vertex groups using direct and manually edited mapping rows"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        scene = context.scene
        settings = scene.mdt_group_match_settings
        source_obj, _target_obj = resolve_group_mapping_objects(settings)

        if source_obj is None:
            self.report({"ERROR"}, "Choose a valid Source mesh.")
            return {"CANCELLED"}
        if len(scene.mdt_group_match_results) == 0:
            self.report({"ERROR"}, "Analyze Mapping first.")
            return {"CANCELLED"}

        result = apply_group_mapping_rename(source_obj, scene.mdt_group_match_results)
        if result["renamed_count"] == 0:
            self.report({"WARNING"}, "No direct or manual mapping rows could be renamed.")
            return {"CANCELLED"}

        message = f"Renamed {result['renamed_count']} vertex group(s)."
        if result["skipped_conflict_count"] > 0:
            message += f" Skipped {result['skipped_conflict_count']} conflicting row(s)."
        if result["skipped_missing_count"] > 0:
            message += f" Skipped {result['skipped_missing_count']} missing source row(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_accept_group_match_candidates(bpy.types.Operator):
    """Promote every current best candidate into an applicable manual mapping."""

    bl_idname = "mdt.accept_group_match_candidates"
    bl_label = "Accept Best For All"
    bl_description = "Turn every row with a suggested target into an applicable mapping for bulk rename"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mdt_group_match_results) > 0

    def execute(self, context):
        scene = context.scene
        promoted_count = 0
        skipped_count = 0

        for item in scene.mdt_group_match_results:
            if not item.source_group or not item.target_group:
                skipped_count += 1
                continue
            if item.status in {"DIRECT", "MANUAL"}:
                continue

            item.status = "MANUAL"
            item.confidence = 0.0
            item.score_overlap = 0.0
            item.score_centroid = 0.0
            item.reason = "Accepted current best candidate for bulk rename."
            promoted_count += 1

        applicable_count = count_applicable_group_matches(scene)
        if promoted_count == 0:
            if applicable_count > 0:
                self.report({"INFO"}, f"{applicable_count} mapping row(s) are already ready to apply.")
                return {"CANCELLED"}
            self.report({"WARNING"}, f"No rows with usable target candidates were found. Skipped {skipped_count}.")
            return {"CANCELLED"}

        message = f"Accepted {promoted_count} mapping row(s) for bulk rename."
        if skipped_count > 0:
            message += f" Skipped {skipped_count} row(s) without a target candidate."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_add_group_match_row(bpy.types.Operator):
    """Add one manual vertex-group mapping row."""

    bl_idname = "mdt.add_group_match_row"
    bl_label = "Add Mapping"
    bl_description = "Add one manual vertex-group mapping row"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        scene = context.scene
        item = scene.mdt_group_match_results.add()
        item.status = "MANUAL"
        item.reason = "Manual mapping."
        scene.mdt_active_group_match_result_index = len(scene.mdt_group_match_results) - 1
        self.report({"INFO"}, "Added one manual mapping row.")
        return {"FINISHED"}


class MDT_OT_remove_group_match_row(bpy.types.Operator):
    """Remove the selected vertex-group mapping row."""

    bl_idname = "mdt.remove_group_match_row"
    bl_label = "Remove Mapping"
    bl_description = "Remove the selected vertex-group mapping row"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mdt_group_match_results) > 0

    def execute(self, context):
        scene = context.scene
        index = scene.mdt_active_group_match_result_index
        if index < 0 or index >= len(scene.mdt_group_match_results):
            self.report({"ERROR"}, "Choose a mapping row to remove.")
            return {"CANCELLED"}

        scene.mdt_group_match_results.remove(index)
        if len(scene.mdt_group_match_results) == 0:
            scene.mdt_active_group_match_result_index = 0
        else:
            scene.mdt_active_group_match_result_index = min(index, len(scene.mdt_group_match_results) - 1)
        self.report({"INFO"}, "Removed one mapping row.")
        return {"FINISHED"}
