"""Operators for selected-bone armature merging."""

import bpy

from ..core.bone_merge import (
    analyze_bone_merge,
    apply_bone_merge,
    collect_selected_source_bone_names,
    parse_bone_names,
    resolve_bone_merge_armatures,
    serialize_bone_names,
)
from ..core.common import ensure_object_mode, is_armature_object, restore_context_mode


def _summarize_analysis(result):
    message = (
        f"Captured {result['source_count']} source bone(s): "
        f"{result['updated_count']} update, {result['added_count']} add."
    )
    if result["missing_source_count"] > 0:
        message += f" Missing source: {result['missing_source_count']}."
    if result["missing_parent_count"] > 0:
        message += f" Missing parent anchor: {result['missing_parent_count']}."
    if result["constraints_on_updated_count"] > 0:
        message += f" Target constraints on updated bones: {result['constraints_on_updated_count']}."
    if result["constraints_targeting_updated_count"] > 0:
        message += f" Constraints targeting updated bones: {result['constraints_targeting_updated_count']}."
    if result["affected_ik_count"] > 0:
        message += f" IK chains affected: {result['affected_ik_count']}."
    if result["invalid_subtarget_count"] > 0:
        message += f" Invalid constraint subtargets: {result['invalid_subtarget_count']}."
    return message


class MDT_OT_capture_bone_merge_selection(bpy.types.Operator):
    """Capture selected bones from the source armature."""

    bl_idname = "mdt.capture_bone_merge_selection"
    bl_label = "Capture Selected Bones"
    bl_description = "Capture selected source armature bones for body rest-bone transfer"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_bone_merge_settings
        source_obj, _target_obj = resolve_bone_merge_armatures(context, settings)
        if not is_armature_object(source_obj):
            self.report({"ERROR"}, "Choose a valid source armature.")
            return {"CANCELLED"}

        try:
            bone_names = collect_selected_source_bone_names(context, source_obj)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        settings.source_object = source_obj
        settings.captured_bone_names = serialize_bone_names(bone_names)
        self.report({"INFO"}, f"Captured {len(bone_names)} source bone(s) from {source_obj.name}.")
        return {"FINISHED"}


class MDT_OT_analyze_bone_merge(bpy.types.Operator):
    """Analyze selected source bones before applying them to the target armature."""

    bl_idname = "mdt.analyze_bone_merge"
    bl_label = "Analyze Bone Merge"
    bl_description = "Preview how captured source bones will update or add bones on the target armature"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_bone_merge_settings
        source_obj, target_obj = resolve_bone_merge_armatures(context, settings)
        bone_names = parse_bone_names(settings.captured_bone_names)

        try:
            result = analyze_bone_merge(source_obj, target_obj, bone_names)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, _summarize_analysis(result))
        return {"FINISHED"}


class MDT_OT_apply_bone_merge(bpy.types.Operator):
    """Apply captured source rest bones into the target armature."""

    bl_idname = "mdt.apply_bone_merge"
    bl_label = "Apply Bone Merge"
    bl_description = "Replace target rest bones with captured source bones and add missing source bones"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        settings = context.scene.mdt_bone_merge_settings
        source_obj, target_obj = resolve_bone_merge_armatures(context, settings)
        bone_names = parse_bone_names(settings.captured_bone_names)

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = apply_bone_merge(
                context,
                source_obj,
                target_obj,
                bone_names,
                use_source_parenting=settings.use_source_parenting,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        message = (
            f"Applied {result['source_count']} source bone(s) to {target_obj.name}: "
            f"{result['updated_count']} updated, {result['added_count']} added. "
            "Target constraints kept."
        )
        if result["reparented_count"] > 0:
            message += f" Reparented {result['reparented_count']} bone(s)."
        if result["missing_parent_count"] > 0:
            message += f" Missing parent anchor: {result['missing_parent_count']}."
        if result["constraints_on_updated_count"] > 0 or result["affected_ik_count"] > 0:
            message += (
                f" Constraint risk: {result['constraints_on_updated_count']} on updated bones, "
                f"{result['affected_ik_count']} IK chain(s)."
            )
        self.report({"INFO"}, message)
        return {"FINISHED"}
