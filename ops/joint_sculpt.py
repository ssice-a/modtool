"""Session controls for joint sculpting; native Blender handles every brush stroke."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.joint_sculpt import apply_joint_sculpt, cancel_joint_sculpt, start_joint_sculpt


class MDT_OT_start_joint_sculpt(bpy.types.Operator):
    bl_idname = "mdt.start_joint_sculpt"
    bl_label = "Start Joint Sculpt"
    bl_description = "Sculpt selected meshes together on a temporary union; originals retain their UVs, materials and weights"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_sculpt_settings
        return not settings.sources and settings.proxy is None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            result = start_joint_sculpt(context, get_selected_mesh_objects(context), previous)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Joint Sculpt: {result['object_count']} objects, {result['vertex_count']} vertices. Keep topology unchanged, then Apply or Cancel.")
        return {"FINISHED"}


class MDT_OT_apply_joint_sculpt(bpy.types.Operator):
    bl_idname = "mdt.apply_joint_sculpt"
    bl_label = "Apply Joint Sculpt"
    bl_description = "Write vertex displacements back to every original mesh and all its shape keys, preserving shape-key differences"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_sculpt_settings
        return bool(settings.sources) and settings.proxy is not None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            result = apply_joint_sculpt(context)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Updated {result['changed_vertices']} vertices in {result['object_count']} objects.")
        return {"FINISHED"}


class MDT_OT_cancel_joint_sculpt(bpy.types.Operator):
    bl_idname = "mdt.cancel_joint_sculpt"
    bl_label = "Cancel Joint Sculpt"
    bl_description = "Discard the temporary sculpt and restore the original objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_sculpt_settings
        return bool(settings.sources) or settings.proxy is not None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            cancel_joint_sculpt(context)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}
