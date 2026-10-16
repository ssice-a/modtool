"""Import MMD body poses and motions into a selected target rig."""

import bpy
from bpy_extras.io_utils import ImportHelper

from ..core.common import is_armature_object
from ..core.mmd_animation import import_mmd_motion


class MDT_OT_import_mmd_animation(bpy.types.Operator, ImportHelper):
    bl_idname = "mdt.import_mmd_animation"
    bl_label = "Import MMD Motion / Pose"
    bl_description = "Choose a VMD or VPD file and import it onto the chosen FK/game body skeleton"
    bl_options = {"REGISTER", "UNDO"}

    filter_glob: bpy.props.StringProperty(default="*.vmd;*.vpd", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "POSE"}

    def execute(self, context):
        settings = context.scene.mdt_animation_settings
        rig = settings.target_armature
        if not is_armature_object(rig):
            self.report({"ERROR"}, "Choose a target armature in Animation.")
            return {"CANCELLED"}
        path = self.filepath
        try:
            result = import_mmd_motion(context, rig, path,
                                       translation_multiplier=settings.translation_multiplier,
                                       start_frame=settings.start_frame)
        except (ValueError, RuntimeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        settings.last_action = result["action"]
        skipped = len(result["ignored"]) + len(result["missing"])
        messages = []
        if skipped:
            messages.append(f"unmapped controls: {skipped}")
        if result["morph_names"]:
            messages.append(f"unmapped expressions: {len(result['morph_names'])}")
        if result["ik_clipped_samples"]:
            messages.append(f"unreachable leg IK samples: {result['ik_clipped_samples']}")
        if result["untranslated"]:
            messages.append(f"unsupported local translations: {len(result['untranslated'])}")
        settings.last_report = "; ".join(messages) or "Standard body controls imported."
        self.report({"WARNING"} if messages else {"INFO"}, f"{result['action']}: {len(result['frames'])} frame(s). {settings.last_report}")
        return {"FINISHED"}
