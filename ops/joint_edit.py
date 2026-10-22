"""Session controls for joint editing; native Blender handles every brush stroke."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.joint_edit import apply_joint_edit, cancel_joint_edit, start_joint_edit


class MDT_OT_start_joint_edit(bpy.types.Operator):
    bl_idname = "mdt.start_joint_edit"
    bl_label = "Start Joint Edit"
    bl_description = "Sculpt or weight-paint selected meshes together; Apply writes edits back to the separate originals"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_edit_settings
        return not settings.sources and settings.proxy is None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            result = start_joint_edit(context, get_selected_mesh_objects(context), previous)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Joint Edit: {result['object_count']} objects, {result['vertex_count']} vertices. Keep topology unchanged, then Apply or Cancel.")
        return {"FINISHED"}


class MDT_OT_apply_joint_edit(bpy.types.Operator):
    bl_idname = "mdt.apply_joint_edit"
    bl_label = "Apply Joint Edit"
    bl_description = "Apply geometry and painted weights to the separate originals, preserving UVs and shape-key differences"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_edit_settings
        return bool(settings.sources) and settings.proxy is not None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            result = apply_joint_edit(context)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Updated {result['changed_vertices']} positions and {result['painted_vertices']} painted vertices in {result['object_count']} objects.")
        return {"FINISHED"}


class MDT_OT_cancel_joint_edit(bpy.types.Operator):
    bl_idname = "mdt.cancel_joint_edit"
    bl_label = "Cancel Joint Edit"
    bl_description = "Discard geometry and weight edits and restore the original objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_edit_settings
        return bool(settings.sources) or settings.proxy is not None

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            cancel_joint_edit(context)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class MDT_OT_switch_joint_edit_mode(bpy.types.Operator):
    bl_idname = "mdt.switch_joint_edit_mode"
    bl_label = "Switch Joint Edit Mode"
    bl_description = "Switch the joint editing mesh between sculpting and weight painting"
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(items=(("SCULPT", "Sculpt", ""), ("WEIGHT_PAINT", "Weight Paint", "")))

    @classmethod
    def poll(cls, context):
        settings = context.scene.mdt_joint_edit_settings
        return bool(settings.sources) and settings.proxy is not None

    def execute(self, context):
        settings = context.scene.mdt_joint_edit_settings
        previous = context.mode
        try:
            ensure_object_mode(context)
            for obj in list(context.selected_objects):
                obj.select_set(False)
            settings.proxy.select_set(True)
            context.view_layer.objects.active = settings.proxy
            bpy.ops.object.mode_set(mode=self.mode)
            settings.mode = self.mode
        except RuntimeError as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}
