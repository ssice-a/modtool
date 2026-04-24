"""Seam operators for Mod Tool."""

import bpy

from ..core.common import ensure_object_mode, is_mesh_object, restore_context_mode
from ..core.seam import (
    apply_weights_to_vertex,
    build_mapping_candidates,
    build_vertex_pairs,
    collect_seam_vertices,
    resolve_apply_objects,
    resolve_enabled_mapping_direction,
    resolve_mapping_objects,
    resolve_syncable_group_names,
    store_group_mappings,
)


class MDT_OT_analyze_seam_mapping(bpy.types.Operator):
    """Analyze nearby seam vertex pairs and infer cross-object group mappings."""

    bl_idname = "mdt.analyze_seam_mapping"
    bl_label = "Analyze Mapping"
    bl_description = "Pair nearby seam vertices and infer which vertex groups correspond across both objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_seam_settings
        source_obj, target_obj = resolve_mapping_objects(context, settings)

        if not is_mesh_object(source_obj):
            self.report({"ERROR"}, "Choose a valid analysis source mesh.")
            return {"CANCELLED"}

        if not is_mesh_object(target_obj):
            self.report({"ERROR"}, "Choose a valid analysis target mesh.")
            return {"CANCELLED"}

        if source_obj == target_obj:
            self.report({"ERROR"}, "Source and target objects must be different.")
            return {"CANCELLED"}

        source_seam_vertices = collect_seam_vertices(source_obj)
        target_seam_vertices = collect_seam_vertices(target_obj)

        if not source_seam_vertices:
            store_group_mappings(context.scene, [])
            self.report({"ERROR"}, "The source object has no open-boundary seam vertices.")
            return {"CANCELLED"}

        if not target_seam_vertices:
            store_group_mappings(context.scene, [])
            self.report({"ERROR"}, "The target object has no open-boundary seam vertices.")
            return {"CANCELLED"}

        matched_pairs, unmatched_source, unmatched_target = build_vertex_pairs(
            source_seam_vertices,
            target_seam_vertices,
            settings.match_tolerance,
        )

        if not matched_pairs:
            store_group_mappings(context.scene, [])
            self.report({"ERROR"}, "No nearby seam vertex pairs were found. Increase Match Tolerance.")
            return {"CANCELLED"}

        syncable_group_names = resolve_syncable_group_names(
            source_obj,
            target_obj,
            settings.only_deform_groups,
        )
        mappings = build_mapping_candidates(
            source_obj,
            target_obj,
            matched_pairs,
            syncable_group_names,
            settings.weight_tolerance,
        )
        store_group_mappings(context.scene, mappings)

        if not mappings:
            self.report(
                {"ERROR"},
                "No group mappings were inferred from the seam pairs. Increase Weight Tolerance.",
            )
            return {"CANCELLED"}

        message = f"Stored {len(mappings)} mapping(s) from {len(matched_pairs)} seam pair(s)."
        if unmatched_source or unmatched_target:
            message += f" Unmatched: {len(unmatched_source)} source, {len(unmatched_target)} target."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class MDT_OT_clear_group_mappings(bpy.types.Operator):
    """Clear stored group mappings."""

    bl_idname = "mdt.clear_group_mappings"
    bl_label = "Clear Mapping"
    bl_description = "Clear the currently stored seam mapping results"

    def execute(self, context):
        store_group_mappings(context.scene, [])
        self.report({"INFO"}, "Stored mappings were cleared.")
        return {"FINISHED"}


class MDT_OT_apply_seam_mapping(bpy.types.Operator):
    """Copy source seam weights to the target seam through stored group mappings."""

    bl_idname = "mdt.apply_seam_mapping"
    bl_label = "Apply To Target"
    bl_description = "Pair nearby seam vertices and write mapped source weights onto the target seam"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_seam_settings
        source_obj, target_obj = resolve_apply_objects(context, settings)

        if not is_mesh_object(source_obj):
            self.report({"ERROR"}, "Choose a valid apply source mesh.")
            return {"CANCELLED"}

        if not is_mesh_object(target_obj):
            self.report({"ERROR"}, "Choose a valid apply target mesh.")
            return {"CANCELLED"}

        if source_obj == target_obj:
            self.report({"ERROR"}, "Source and target objects must be different.")
            return {"CANCELLED"}

        resolved_mappings, skipped_missing_source_groups = resolve_enabled_mapping_direction(
            context.scene,
            source_obj,
            settings.invert_mapping,
        )
        if not resolved_mappings:
            self.report(
                {"ERROR"},
                "No enabled mappings fit the current apply direction.",
            )
            return {"CANCELLED"}

        source_seam_vertices = collect_seam_vertices(source_obj)
        target_seam_vertices = collect_seam_vertices(target_obj)

        if not source_seam_vertices:
            self.report({"ERROR"}, "The source object has no open-boundary seam vertices.")
            return {"CANCELLED"}

        if not target_seam_vertices:
            self.report({"ERROR"}, "The target object has no open-boundary seam vertices.")
            return {"CANCELLED"}

        matched_pairs, unmatched_source, unmatched_target = build_vertex_pairs(
            source_seam_vertices,
            target_seam_vertices,
            settings.match_tolerance,
        )

        if not matched_pairs:
            self.report({"ERROR"}, "No nearby seam vertex pairs were found. Increase Match Tolerance.")
            return {"CANCELLED"}

        syncable_group_names = resolve_syncable_group_names(
            source_obj,
            target_obj,
            settings.only_deform_groups,
        )
        if syncable_group_names is not None:
            resolved_mappings = [
                (source_group, target_group)
                for source_group, target_group in resolved_mappings
                if source_group in syncable_group_names and target_group in syncable_group_names
            ]
            if not resolved_mappings:
                self.report(
                    {"ERROR"},
                    "No enabled mappings remain after filtering to deform groups.",
                )
                return {"CANCELLED"}

        target_groups_to_clear = {target_group for _source_group, target_group in resolved_mappings}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            for source_index, target_index in matched_pairs:
                apply_weights_to_vertex(
                    source_obj,
                    source_index,
                    target_obj,
                    target_index,
                    resolved_mappings,
                    target_groups_to_clear,
                    syncable_group_names,
                )
        finally:
            restore_context_mode(previous_mode)

        message = f"Applied {len(resolved_mappings)} mapping(s) across {len(matched_pairs)} seam pair(s)."
        if skipped_missing_source_groups:
            message += f" Skipped {skipped_missing_source_groups} missing source mapping(s)."
        if unmatched_source or unmatched_target:
            message += f" Unmatched: {len(unmatched_source)} source, {len(unmatched_target)} target."
        self.report({"INFO"}, message)
        return {"FINISHED"}
