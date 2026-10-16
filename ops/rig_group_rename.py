"""Operator for active-mesh game-rig vertex-group conversion."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, restore_context_mode
from ..core.rig_group_rename import (
    AUTO_RIG_NPC_MAPPING_ID,
    AUTO_RIG_ZMD_MAPPING_ID,
    apply_auto_rig_npc_batch,
    rig_mapping_items,
)


class MDT_OT_auto_rig_npc(bpy.types.Operator):
    """Convert selected mesh groups using the selected offline mapping."""

    bl_idname = "mdt.auto_rig_npc"
    bl_label = "Rename Selected"
    bl_description = "Rename selected mesh vertex groups using the target armature's bone names"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(
            len(getattr(obj, "vertex_groups", ())) > 0
            for obj in get_selected_mesh_objects(context)
        )

    def execute(self, context):
        settings = context.scene.mdt_group_match_settings
        mapping_id = settings.rig_mapping
        valid_mapping_ids = {item[0] for item in rig_mapping_items(None, None)}
        if mapping_id not in valid_mapping_ids:
            mapping_id = AUTO_RIG_NPC_MAPPING_ID
            settings.rig_mapping = mapping_id
        target_armature = settings.target_armature
        if mapping_id == AUTO_RIG_ZMD_MAPPING_ID and target_armature is None:
            self.report({"ERROR"}, "Choose a target armature before using autorig->zmd.")
            return {"CANCELLED"}
        target_bone_names = (
            {bone.name for bone in target_armature.data.bones}
            if target_armature is not None
            else None
        )
        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        try:
            source_objects = [
                obj
                for obj in get_selected_mesh_objects(context)
                if len(obj.vertex_groups) > 0
            ]
            if not source_objects:
                self.report({"ERROR"}, "Select at least one mesh with vertex groups.")
                return {"CANCELLED"}
            results = apply_auto_rig_npc_batch(
                source_objects,
                mapping_id,
                target_bone_names=target_bone_names,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            restore_context_mode(previous_mode)

        renamed_group_count = sum(result["created_count"] for _obj, result in results)
        removed_group_count = sum(result["removed_group_count"] for _obj, result in results)
        merged_target_count = sum(result["merge_target_count"] for _obj, result in results)
        normalized_vertex_count = sum(result["normalized_vertex_count"] for _obj, result in results)
        self.report(
            {"INFO"},
            (
                f"Processed {len(results)} mesh(es), renamed {renamed_group_count} "
                f"group(s), removed {removed_group_count} unmatched group(s), "
                f"merged {merged_target_count} target group(s), normalized "
                f"{normalized_vertex_count} vertex/vertices."
            ),
        )
        return {"FINISHED"}
