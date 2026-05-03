"""Armature cleanup operators for Mod Tool."""

import bpy

from ..core.armature_cleanup import (
    clean_fk_armature,
    get_cleanup_mesh_objects,
    resolve_cleanup_armature,
)
from ..core.common import ensure_object_mode, restore_context_mode


class MDT_OT_clean_fk_armature(bpy.types.Operator):
    """Convert an armature into a simple weighted FK rig."""

    bl_idname = "mdt.clean_fk_armature"
    bl_label = "Clean To FK"
    bl_description = (
        "Keep bones that match weighted vertex groups, remove unweighted/control bones, "
        "strip pose constraints and transform locks, and preserve the retained parent hierarchy"
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

        mesh_objects = get_cleanup_mesh_objects(context, armature_obj)
        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            result = clean_fk_armature(context, armature_obj, mesh_objects)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        self.report(
            {"INFO"},
            (
                f"Cleaned {armature_obj.name}: kept {result['kept_bone_count']} / "
                f"{result['original_bone_count']} bone(s), removed {result['removed_bone_count']} "
                f"bone(s), stripped {result['constraint_count']} pose constraint(s), "
                f"cleared {result['lock_count']} lock(s)."
            ),
        )
        return {"FINISHED"}
