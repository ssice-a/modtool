"""Boundary-loop weld operator."""

import bpy

from ..core.bridge_loops import bridge_selected_loops
from ..core.common import is_mesh_object


class MDT_OT_bridge_uv_loops(bpy.types.Operator):
    """Weld the non-active boundary loop onto the active target loop."""

    bl_idname = "mdt.bridge_uv_loops"
    bl_label = "Bridge and Weld Loops"
    bl_description = (
        "Weld the non-active selected boundary loop to the active loop, preserving "
        "the target mesh, UV layers, shape keys, and vertex data"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and is_mesh_object(context.view_layer.objects.active)

    def execute(self, context):
        try:
            result = bridge_selected_loops(context)
        except (RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"Welded {result['welded_vertex_count']} seam vertices; inserted "
                f"{result['inserted_vertex_count']} matching vertices and created no faces. "
                f"Max seam gap {result['max_seam_distance']:.6g}; max quad warp "
                f"{result['max_quad_warp_degrees']:.1f} degrees."
            ),
        )
        return {"FINISHED"}
