"""Shape-key transfer operator for Mod Tool."""

import bpy

from ..core.common import ensure_object_mode, is_mesh_object, restore_context_mode
from ..core.shape_keys import (
    build_surface_samples,
    build_vertex_samples,
    has_transferable_shape_keys,
    resolve_transfer_objects,
    transfer_shape_keys,
    transfer_shape_keys_by_vertex,
    transfer_shape_keys_by_wrap,
)


class MDT_OT_transfer_shape_keys(bpy.types.Operator):
    """Transfer all source shape keys onto another mesh."""

    bl_idname = "mdt.transfer_shape_keys"
    bl_label = "Transfer A -> B"
    bl_description = "Transfer every non-basis shape key from the source mesh onto the target mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_shape_key_settings
        source_obj, target_obj = resolve_transfer_objects(context, settings)

        if not is_mesh_object(source_obj):
            self.report({"ERROR"}, "Choose a valid source mesh with shape keys.")
            return {"CANCELLED"}

        if not has_transferable_shape_keys(source_obj):
            self.report({"ERROR"}, "The source mesh has no transferable shape keys.")
            return {"CANCELLED"}

        if not is_mesh_object(target_obj):
            self.report({"ERROR"}, "Choose a valid target mesh.")
            return {"CANCELLED"}

        if source_obj == target_obj:
            self.report({"ERROR"}, "Source and target objects must be different.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            if settings.transfer_mode == "WRAP":
                transfer_result = transfer_shape_keys_by_wrap(
                    context,
                    source_obj,
                    target_obj,
                    replace_existing=settings.replace_existing,
                )
            elif settings.transfer_mode == "VERTEX":
                vertex_data = build_vertex_samples(
                    source_obj,
                    target_obj,
                    settings.vertex_tolerance,
                    settings.shared_vertex_tolerance,
                )
                if vertex_data["unmatched_count"] > 0:
                    self.report(
                        {"ERROR"},
                        "Target vertices could not be matched by nearby source vertices or face hints. "
                        f"Unmatched: {vertex_data['unmatched_count']} / {vertex_data['target_vertex_count']}. "
                        f"Nearest miss: {vertex_data['farthest_distance']:.6f}. Increase Vertex Tolerance if needed.",
                    )
                    return {"CANCELLED"}

                transfer_result = transfer_shape_keys_by_vertex(
                    source_obj,
                    target_obj,
                    vertex_data["samples"],
                    replace_existing=settings.replace_existing,
                )
            else:
                sample_data = build_surface_samples(
                    source_obj,
                    target_obj,
                    settings.max_distance,
                    settings.shared_vertex_tolerance,
                )
                if sample_data["unmatched_count"] > 0:
                    self.report(
                        {"ERROR"},
                        "Target vertices did not fully match the source surface. "
                        f"Unmatched: {sample_data['unmatched_count']} / {sample_data['target_vertex_count']}. "
                        f"Farthest hit: {sample_data['farthest_distance']:.6f}. Increase Surface Tolerance if needed.",
                    )
                    return {"CANCELLED"}

                transfer_result = transfer_shape_keys(
                    source_obj,
                    target_obj,
                    sample_data["samples"],
                    replace_existing=settings.replace_existing,
                )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        if transfer_result["created_count"] == 0:
            self.report({"WARNING"}, "No shape keys were transferred.")
            return {"CANCELLED"}

        message = f"Transferred {transfer_result['created_count']} shape key(s)."
        if transfer_result["replaced_count"] > 0:
            message += f" Replaced {transfer_result['replaced_count']} existing key(s)."
        if transfer_result["skipped_count"] > 0:
            message += f" Skipped {transfer_result['skipped_count']} same-name key(s)."
        if settings.transfer_mode == "WRAP":
            message += " Wrap baked with Surface Deform."
        elif settings.transfer_mode == "VERTEX":
            if vertex_data["grouped_vertex_count"] > 0:
                message += f" Shared motion across {vertex_data['grouped_vertex_count']} overlapping target vertices."
            if vertex_data["face_guided_group_count"] > 0:
                message += (
                    f" Face hints guided {vertex_data['face_guided_group_count']} target groups."
                )
            if vertex_data["ambiguous_group_count"] > 0:
                message += (
                    f" Ambiguous target groups resolved by hybrid vertex/face scoring: "
                    f"{vertex_data['ambiguous_group_count']}."
                )
            if vertex_data["max_face_corner_distance"] > 0.0:
                message += f" Max face-hint corner offset: {vertex_data['max_face_corner_distance']:.6f}."
            message += f" Target groups: {vertex_data['target_group_count']}."
        else:
            if sample_data["ignored_degenerate_triangle_count"] > 0:
                message += f" Ignored {sample_data['ignored_degenerate_triangle_count']} degenerate source triangle(s)."
            if sample_data["grouped_vertex_count"] > 0:
                message += f" Shared motion across {sample_data['grouped_vertex_count']} overlapping target vertices."
        self.report({"INFO"}, message)
        return {"FINISHED"}
