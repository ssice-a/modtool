"""Sidebar panels for Mod Tool."""

import bpy

from ..core.common import is_mesh_object
from ..core.bone_merge import parse_bone_names, resolve_bone_merge_armatures
from ..core.group_mapping import resolve_group_mapping_objects
from ..core.shape_keys import resolve_transfer_objects


class _MDT_PT_base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mod Tool"


class MDT_PT_vertex_groups(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for vertex-group cleanup and rename tools."""

    bl_label = "Vertex Groups"

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout
        obj = context.object
        scene = context.scene
        if is_mesh_object(obj) and len(obj.vertex_groups) > 0:
            settings = context.scene.mdt_rename_settings
            layout.prop(settings, "mode", expand=True)

            settings_box = layout.box()
            settings_col = settings_box.column(align=True)
            if settings.mode == "MODIFY":
                settings_col.prop(settings, "search")
                settings_col.prop(settings, "replace")
                toggle_row = settings_col.row(align=True)
                toggle_row.prop(settings, "use_regex")
                toggle_row.prop(settings, "case_sensitive")
                name_row = settings_col.row(align=True)
                name_row.prop(settings, "prefix")
                name_row.prop(settings, "suffix")
            else:
                settings_col.prop(settings, "sequence_base")
                settings_col.prop(settings, "sequence_separator")
                sequence_row = settings_col.row(align=True)
                sequence_row.prop(settings, "sequence_start")
                sequence_row.prop(settings, "sequence_padding")

            button_row = layout.row(align=True)
            button_row.operator("mdt.batch_rename_groups", icon="CHECKMARK")
            button_row.operator("mdt.reset_rename_settings", icon="LOOP_BACK")

            cleanup_row = layout.row()
            cleanup_row.operator(
                "mdt.remove_empty_vertex_groups",
                text="Remove Empty",
                icon="TRASH",
            )
            cleanup_row.operator(
                "mdt.clean_disconnected_vertex_group_weights",
                text="Clean Islands",
                icon="BRUSH_DATA",
            )
            keep_row = layout.row()
            keep_row.operator(
                "mdt.keep_deforming_vertex_groups",
                text="Keep Bone Groups",
                icon="ARMATURE_DATA",
            )
            body_row = layout.row()
            body_row.operator(
                "mdt.keep_body_vertex_groups",
                text="Keep Body Groups",
                icon="GROUP_VERTEX",
            )
            mirror_row = layout.row(align=True)
            left_to_right = mirror_row.operator(
                "mdt.mirror_vertex_group_weights",
                text="L -> R",
                icon="MOD_MIRROR",
            )
            left_to_right.direction = "LEFT_TO_RIGHT"
            right_to_left = mirror_row.operator(
                "mdt.mirror_vertex_group_weights",
                text="R -> L",
                icon="MOD_MIRROR",
            )
            right_to_left.direction = "RIGHT_TO_LEFT"
        else:
            layout.label(text="Rule rename uses the active mesh object's vertex groups.", icon="INFO")

        seam_row = layout.row()
        seam_row.operator("mdt.rename_selected_seam_groups", icon="MOD_VERTEX_WEIGHT")

        mapping_settings = scene.mdt_group_match_settings
        source_obj, target_obj = resolve_group_mapping_objects(mapping_settings)
        mapping_box = layout.box()
        mapping_box.label(text="Group Mapping", icon="LINKED")

        mapping_col = mapping_box.column(align=True)
        mapping_col.prop(mapping_settings, "source_object")
        mapping_col.prop(mapping_settings, "target_object")

        rename_row = mapping_box.row()
        rename_row.enabled = (
            source_obj is not None
            and target_obj is not None
            and source_obj != target_obj
        )
        rename_row.operator("mdt.rename_groups_from_target", icon="CHECKMARK")

        seam_weight_row = mapping_box.row()
        seam_weight_row.enabled = (
            source_obj is not None
            and target_obj is not None
            and source_obj != target_obj
        )
        seam_weight_row.operator("mdt.unify_seam_weights_from_target", icon="MOD_VERTEX_WEIGHT")

        mapping_box.label(text="Select Source and Target, then rename.", icon="INFO")


class MDT_PT_shape_keys(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for cross-mesh shape-key transfer."""

    bl_label = "Shape Keys"

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mdt_shape_key_settings
        source_obj, target_obj = resolve_transfer_objects(context, settings)

        box = layout.box()
        col = box.column(align=True)
        col.prop(settings, "transfer_mode", expand=True)
        col.prop(settings, "source_object")
        col.prop(settings, "target_object")
        if settings.transfer_mode == "VERTEX":
            col.prop(settings, "shared_vertex_tolerance")
            col.prop(settings, "vertex_tolerance")
        elif settings.transfer_mode == "SURFACE":
            col.prop(settings, "shared_vertex_tolerance")
            col.prop(settings, "max_distance")
        col.prop(settings, "replace_existing")

        action_row = layout.row()
        action_row.enabled = (
            source_obj is not None
            and target_obj is not None
            and source_obj != target_obj
        )
        action_row.operator("mdt.transfer_shape_keys", icon="MOD_MESHDEFORM")

        range_box = layout.box()
        range_box.label(text="Shape Key Ranges", icon="SHAPEKEY_DATA")
        range_row = range_box.row(align=True)
        range_row.operator(
            "mdt.set_shape_key_current_values_as_zero",
            text="Current Values as Zero",
            icon="CHECKMARK",
        )
        range_row.operator(
            "mdt.normalize_shape_key_ranges",
            text="Normalize 0-1",
            icon="CHECKMARK",
        )

        action_row = layout.row()
        action_row.operator("mdt.apply_modifier_keep_shape_keys", text="Apply Enabled Modifiers", icon="CHECKMARK")


class MDT_PT_materials(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for material creation and material-based splitting."""

    bl_label = "Materials"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout

        create_row = layout.row()
        create_row.operator("mdt.create_named_materials", icon="MATERIAL")

        split_row = layout.row()
        split_row.operator("mdt.separate_by_material", icon="MOD_EXPLODE")


class MDT_PT_armature_cleanup(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for armature FK cleanup."""

    bl_label = "Armature"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout

        mirror_settings = context.scene.mdt_pose_mirror_settings
        mirror_box = layout.box()
        mirror_box.prop(mirror_settings, "enabled", toggle=True)

        merge_settings = context.scene.mdt_bone_merge_settings
        source_obj, target_obj = resolve_bone_merge_armatures(context, merge_settings)
        captured_bone_count = len(parse_bone_names(merge_settings.captured_bone_names))

        merge_box = layout.box()
        merge_box.label(text="Bone Merge", icon="ARMATURE_DATA")
        merge_col = merge_box.column(align=True)
        merge_col.prop(merge_settings, "source_object")
        merge_col.prop(merge_settings, "target_object")
        merge_col.prop(merge_settings, "use_source_parenting")
        merge_col.operator("mdt.capture_bone_merge_selection", icon="RESTRICT_SELECT_OFF")
        merge_box.label(text=f"Captured bones: {captured_bone_count}", icon="BONE_DATA")

        merge_row = merge_box.row(align=True)
        merge_row.enabled = source_obj is not None and target_obj is not None and captured_bone_count > 0
        merge_row.operator("mdt.analyze_bone_merge", icon="VIEWZOOM")
        merge_row.operator("mdt.apply_bone_merge", icon="CHECKMARK")

        box = layout.box()
        box.label(text="FK Cleanup", icon="ARMATURE_DATA")
        box.label(text="Removes pose constraints and transform locks.", icon="INFO")
        box.label(text="Keeps all bones, hierarchy, and the current visible pose.", icon="INFO")

        action_row = box.row()
        action_row.operator("mdt.clean_fk_armature", icon="ARMATURE_DATA")
