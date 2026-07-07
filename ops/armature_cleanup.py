"""Armature cleanup operators for Mod Tool."""

import bpy

from ..core.armature_cleanup import (
    clean_fk_armature,
    resolve_cleanup_armature,
)
from ..core.common import ensure_object_mode, restore_context_mode


class MDT_OT_clean_fk_armature(bpy.types.Operator):
    """Remove pose rigging helpers and keep the current pose as FK controls."""

    bl_idname = "mdt.clean_fk_armature"
    bl_label = "Clean To FK"
    bl_description = (
        "Remove pose constraints and locks, preserve every bone and parent link, "
        "and keep the current visible pose as editable FK transforms"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        armature_obj, error = resolve_cleanup_armature(context)
        if armature_obj is None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = clean_fk_armature(context, armature_obj)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        self.report(
            {"INFO"},
            (
                f"Cleaned {armature_obj.name}: preserved {result['bone_count']} bone(s) "
                f"and hierarchy, removed {result['constraint_count']} pose constraint(s), "
                f"cleared {result['lock_count']} lock(s), restored current pose on "
                f"{result['restored_pose_count']} FK bone(s)."
            ),
        )
        return {"FINISHED"}
