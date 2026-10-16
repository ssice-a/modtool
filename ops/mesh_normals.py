"""Mesh normal operators for Mod Tool."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.mesh_normals import (
    DEFAULT_NORMAL_MERGE_DISTANCE,
    NORMAL_MERGE_ACTIVE,
    NORMAL_MERGE_AVERAGE,
    merge_selected_vertex_normals,
)


class _MDT_OT_merge_selected_vertex_normals_base:
    """Merge selected vertices' custom normals by world-space distance."""

    bl_options = {"REGISTER", "UNDO"}
    normal_merge_mode = NORMAL_MERGE_AVERAGE

    merge_distance: bpy.props.FloatProperty(
        name="Distance",
        description="World-space distance used to group selected vertices whose normals should be unified",
        default=DEFAULT_NORMAL_MERGE_DISTANCE,
        min=0.0,
        soft_min=0.000001,
        soft_max=0.01,
        precision=6,
    )

    @classmethod
    def poll(cls, context):
        return bool(get_selected_mesh_objects(context))

    def execute(self, context):
        mesh_objects = get_selected_mesh_objects(context)
        if not mesh_objects:
            self.report({"ERROR"}, "Select at least one mesh object.")
            return {"CANCELLED"}
        active_obj = context.view_layer.objects.active

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = merge_selected_vertex_normals(
                mesh_objects,
                merge_distance=self.merge_distance,
                mode=self.normal_merge_mode,
                active_obj=active_obj,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        if result["merged_cluster_count"] == 0:
            self.report({"WARNING"}, "No selected vertices were close enough to merge normals.")
            return {"CANCELLED"}
        if result["written_loop_count"] == 0:
            self.report({"WARNING"}, "No normals were written. Active mode needs each distance group to include an active-object selected vertex.")
            return {"CANCELLED"}

        skipped_message = ""
        if result["skipped_cluster_count"] > 0:
            skipped_message = f" Skipped {result['skipped_cluster_count']} group(s) without active-object selected normals."
        self.report(
            {"INFO"},
            (
                f"Merged normals for {result['touched_vertex_count']} vertex/vertices in "
                f"{result['merged_cluster_count']} distance group(s), wrote "
                f"{result['written_loop_count']} loop normal(s).{skipped_message}"
            ),
        )
        return {"FINISHED"}


class MDT_OT_merge_selected_vertex_normals_average(
    _MDT_OT_merge_selected_vertex_normals_base,
    bpy.types.Operator,
):
    """Merge selected nearby vertices' normals to their average direction."""

    bl_idname = "mdt.merge_selected_vertex_normals_average"
    bl_label = "Merge Normals Average"
    bl_description = "Unify selected nearby vertices' custom normals to each distance group's average direction"
    normal_merge_mode = NORMAL_MERGE_AVERAGE


class MDT_OT_merge_selected_vertex_normals_active(
    _MDT_OT_merge_selected_vertex_normals_base,
    bpy.types.Operator,
):
    """Merge selected nearby vertices' normals to the active object's direction."""

    bl_idname = "mdt.merge_selected_vertex_normals_active"
    bl_label = "Merge Normals From Active"
    bl_description = "Unify selected nearby vertices' custom normals to the active object's selected normal direction"
    normal_merge_mode = NORMAL_MERGE_ACTIVE
