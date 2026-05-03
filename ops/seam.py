"""Seam operators for Mod Tool."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, is_mesh_object, restore_context_mode
from ..core.seam import collect_seam_vertices, rename_seam_groups_to_target


class MDT_OT_rename_selected_seam_groups(bpy.types.Operator):
    """Rename selected meshes' seam vertex groups to matching names from the active mesh."""

    bl_idname = "mdt.rename_selected_seam_groups"
    bl_label = "Rename Seams To Active"
    bl_description = "Rename seam vertex groups on selected meshes to matching names from the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        active_obj = context.object
        return is_mesh_object(active_obj) and len(get_selected_mesh_objects(context)) > 1

    def execute(self, context):
        target_obj = context.object
        if not is_mesh_object(target_obj):
            self.report({"ERROR"}, "Choose an active mesh object as the naming target.")
            return {"CANCELLED"}

        source_objects = [obj for obj in get_selected_mesh_objects(context) if obj != target_obj]
        if not source_objects:
            self.report({"ERROR"}, "Select at least one more mesh object to rename.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            target_seam_vertices = collect_seam_vertices(target_obj)
            if not target_seam_vertices:
                self.report({"ERROR"}, f"{target_obj.name} has no open-boundary seam vertices.")
                return {"CANCELLED"}

            changed_object_count = 0
            unchanged_object_count = 0
            total_mapping_count = 0
            total_renamed_count = 0
            total_conflict_count = 0
            total_missing_count = 0
            skipped_errors = []

            for source_obj in source_objects:
                try:
                    result = rename_seam_groups_to_target(
                        source_obj,
                        target_obj,
                        target_seam_vertices=target_seam_vertices,
                    )
                except ValueError as exc:
                    skipped_errors.append(f"{source_obj.name}: {exc}")
                    continue

                total_mapping_count += result["mapping_count"]
                total_renamed_count += result["renamed_count"]
                total_conflict_count += result["skipped_conflict_count"]
                total_missing_count += result["skipped_missing_count"]
                if result["renamed_count"] > 0:
                    changed_object_count += 1
                else:
                    unchanged_object_count += 1
        finally:
            restore_context_mode(previous_mode)

        if total_renamed_count == 0:
            message = "No selected seam vertex groups needed renaming."
            if skipped_errors:
                message += f" Skipped {len(skipped_errors)} object(s). First: {skipped_errors[0]}"
            if total_conflict_count > 0:
                message += f" Skipped {total_conflict_count} conflicting group(s)."
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        message = (
            f"Renamed {total_renamed_count} seam vertex group(s) on "
            f"{changed_object_count} object(s) using {target_obj.name}."
        )
        if unchanged_object_count > 0:
            message += f" {unchanged_object_count} object(s) already matched."
        if total_conflict_count > 0:
            message += f" Skipped {total_conflict_count} conflicting group(s)."
        if total_missing_count > 0:
            message += f" Skipped {total_missing_count} missing group(s)."
        if skipped_errors:
            message += f" Skipped {len(skipped_errors)} object(s). First: {skipped_errors[0]}"
        if total_mapping_count > total_renamed_count:
            message += f" Checked {total_mapping_count} seam mapping(s)."
        self.report({"INFO"}, message)
        return {"FINISHED"}
