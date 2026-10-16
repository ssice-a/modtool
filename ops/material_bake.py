"""Output-color baking operators."""

import bpy

from ..core.common import ensure_object_mode, is_mesh_object, restore_context_mode
from ..core.material_bake import bake_output_colors, prepare_bake_materials


class MDT_OT_prepare_color_bake(bpy.types.Operator):
    bl_idname = "mdt.prepare_color_bake"
    bl_label = "Read Materials"
    bl_description = "Read active source UV and used materials; choose each material's destination UV below"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_mesh_object(context.object)

    def execute(self, context):
        previous = ensure_object_mode(context)
        try:
            prepare_bake_materials(context.object, context.scene.mdt_color_bake_settings)
        except (ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous)
        return {"FINISHED"}


class MDT_OT_bake_output_colors(bpy.types.Operator):
    bl_idname = "mdt.bake_output_colors"
    bl_label = "Bake Output Colors"
    bl_description = "Bake color chains connected directly to Material Output into one PNG, using per-material destination UVs"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_mesh_object(context.object)

    def execute(self, context):
        previous = ensure_object_mode(context)
        try:
            result = bake_output_colors(context, context.object, context.scene.mdt_color_bake_settings)
        except (ValueError, RuntimeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous)
        if result["degenerate_triangles"]:
            self.report({"WARNING"}, f"{result['degenerate_triangles']} zero-area UV triangles cannot receive image pixels.")
        self.report({"INFO"}, f"Baked {result['material_count']} materials: {result['filepath']}; combined UV: {result['uv_name']}.")
        return {"FINISHED"}
