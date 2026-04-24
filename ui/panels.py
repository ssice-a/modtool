"""Sidebar panels for Mod Tool."""

import bpy

from ..core.common import get_selected_mesh_objects, is_mesh_object
from ..core.group_mapping import count_applicable_group_matches, count_group_match_status, resolve_group_mapping_objects
from ..core.modifier_apply import get_enabled_modifier_names, has_shape_keys as has_modifier_shape_keys
from ..core.rename import count_empty_vertex_groups, get_rename_preview
from ..core.seam import count_enabled_group_mappings, resolve_apply_objects, resolve_mapping_objects
from ..core.shape_keys import has_transferable_shape_keys, resolve_transfer_objects


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

            preview_data = get_rename_preview(obj, settings)
            preview_box = layout.box()
            preview_box.label(
                text=f"Preview {preview_data['changed_count']}/{preview_data['total_count']}",
                icon="HIDE_OFF",
            )

            if preview_data["error"]:
                preview_box.label(text=preview_data["error"], icon="ERROR")
            elif preview_data["changed_count"] == 0:
                preview_box.label(text="No changes yet.", icon="INFO")
            else:
                for old_name, new_name, _changed in preview_data["rows"]:
                    row = preview_box.row(align=True)
                    row.label(text=old_name, icon="GROUP_VERTEX")
                    row.label(text="->")
                    row.label(text=new_name)

                if preview_data["changed_count"] > len(preview_data["rows"]):
                    preview_box.label(
                        text=f"Showing {len(preview_data['rows'])} of {preview_data['changed_count']}",
                        icon="INFO",
                    )

                if preview_data["adjusted_count"] > 0:
                    preview_box.label(text=f"Auto suffixes: {preview_data['adjusted_count']}", icon="SORTALPHA")

            button_row = layout.row(align=True)
            button_row.operator("mdt.batch_rename_groups", icon="CHECKMARK")
            button_row.operator("mdt.reset_rename_settings", icon="LOOP_BACK")

            cleanup_row = layout.row()
            cleanup_row.operator(
                "mdt.remove_empty_vertex_groups",
                text=f"Remove Empty ({count_empty_vertex_groups(obj)})",
                icon="TRASH",
            )
        else:
            layout.label(text="Rule rename uses the active mesh object's vertex groups.", icon="INFO")

        mapping_settings = scene.mdt_group_match_settings
        source_obj, target_obj = resolve_group_mapping_objects(mapping_settings)
        mapping_box = layout.box()
        mapping_box.label(text="Group Mapping", icon="LINKED")

        mapping_col = mapping_box.column(align=True)
        mapping_col.prop(mapping_settings, "source_object")
        mapping_col.prop(mapping_settings, "target_object")
        mapping_col.prop(mapping_settings, "candidate_limit")
        mapping_col.prop(mapping_settings, "surface_tolerance")
        mapping_col.prop(mapping_settings, "prefer_deform_groups")

        analyze_row = mapping_box.row()
        analyze_row.enabled = (
            source_obj is not None
            and target_obj is not None
            and source_obj != target_obj
        )
        analyze_row.operator("mdt.analyze_group_mapping", icon="VIEWZOOM")

        if len(scene.mdt_group_match_results) > 0:
            stats_row = mapping_box.row(align=True)
            stats_row.label(text=f"Direct {count_group_match_status(scene, 'DIRECT')}", icon="CHECKMARK")
            stats_row.label(text=f"Manual {count_group_match_status(scene, 'MANUAL')}", icon="GREASEPENCIL")
            stats_row.label(text=f"Ambiguous {count_group_match_status(scene, 'AMBIGUOUS')}", icon="QUESTION")
            stats_row.label(text=f"Unmatched {count_group_match_status(scene, 'UNMATCHED')}", icon="ERROR")
            stats_row.label(text=f"Conflict {count_group_match_status(scene, 'CONFLICT')}", icon="CANCEL")

            list_row = mapping_box.row()
            list_row.template_list(
                "MDT_UL_group_match_results",
                "",
                scene,
                "mdt_group_match_results",
                scene,
                "mdt_active_group_match_result_index",
                rows=min(8, max(4, len(scene.mdt_group_match_results))),
            )
            button_col = list_row.column(align=True)
            button_col.operator("mdt.add_group_match_row", text="", icon="ADD")
            button_col.operator("mdt.remove_group_match_row", text="", icon="REMOVE")

            if 0 <= scene.mdt_active_group_match_result_index < len(scene.mdt_group_match_results):
                active_item = scene.mdt_group_match_results[scene.mdt_active_group_match_result_index]
                detail_box = mapping_box.box()
                if source_obj is not None:
                    detail_box.prop_search(active_item, "source_group", source_obj, "vertex_groups", text="Source")
                else:
                    detail_box.prop(active_item, "source_group")
                if target_obj is not None:
                    detail_box.prop_search(active_item, "target_group", target_obj, "vertex_groups", text="Target")
                else:
                    detail_box.prop(active_item, "target_group")
                detail_box.label(text=f"Status: {active_item.status.title()} | Confidence: {active_item.confidence:.2f}")
                detail_box.label(
                    text=f"Overlap: {active_item.score_overlap:.2f} | Centroid: {active_item.score_centroid:.2f}"
                )
                detail_box.label(text=active_item.reason or "No extra details.", icon="INFO")
        else:
            mapping_box.label(text="No mapping analysis yet.", icon="INFO")
            button_row = mapping_box.row(align=True)
            button_row.operator("mdt.add_group_match_row", text="Add Manual Row", icon="ADD")

        action_row = mapping_box.row(align=True)
        action_row.enabled = len(scene.mdt_group_match_results) > 0
        action_row.operator("mdt.accept_group_match_candidates", icon="CHECKBOX_HLT")

        apply_row = mapping_box.row()
        apply_row.enabled = (
            source_obj is not None
            and len(scene.mdt_group_match_results) > 0
            and count_applicable_group_matches(scene) > 0
        )
        apply_row.operator("mdt.apply_group_mapping_rename", icon="CHECKMARK")


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
        selected_mesh_objects = get_selected_mesh_objects(context)
        modifier_apply_objects = [
            obj
            for obj in selected_mesh_objects
            if has_modifier_shape_keys(obj) and len(get_enabled_modifier_names(obj)) > 0
        ]

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

        if source_obj is not None and has_transferable_shape_keys(source_obj):
            key_count = len(source_obj.data.shape_keys.key_blocks) - 1
            box.label(text=f"Source Keys: {key_count}", icon="SHAPEKEY_DATA")
        elif source_obj is not None:
            box.label(text="Source has no transferable keys.", icon="INFO")

        action_row = layout.row()
        action_row.enabled = (
            source_obj is not None
            and target_obj is not None
            and source_obj != target_obj
            and has_transferable_shape_keys(source_obj)
        )
        action_row.operator("mdt.transfer_shape_keys", icon="MOD_MESHDEFORM")

        action_row = layout.row()
        action_row.enabled = len(modifier_apply_objects) > 0
        action_row.operator("mdt.apply_modifier_keep_shape_keys", text="Apply Selected Modifiers", icon="CHECKMARK")


class MDT_PT_seam_transfer(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for seam mapping and seam-weight transfer."""

    bl_label = "Seam Transfer"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.mdt_seam_settings
        analyze_source, analyze_target = resolve_mapping_objects(context, settings)
        apply_source, apply_target = resolve_apply_objects(context, settings)

        analyze_box = layout.box()
        analyze_col = analyze_box.column(align=True)
        analyze_col.prop(settings, "source_object")
        analyze_col.prop(settings, "target_object")
        tolerance_row = analyze_col.row(align=True)
        tolerance_row.prop(settings, "match_tolerance")
        tolerance_row.prop(settings, "weight_tolerance")
        analyze_col.prop(settings, "only_deform_groups")
        analyze_row = analyze_box.row(align=True)
        analyze_row.enabled = (
            analyze_source is not None
            and analyze_target is not None
            and analyze_source != analyze_target
        )
        analyze_row.operator("mdt.analyze_seam_mapping", icon="VIEWZOOM")
        analyze_row.operator("mdt.clear_group_mappings", icon="TRASH")

        mapping_box = layout.box()
        mapping_box.label(
            text=f"Mappings {count_enabled_group_mappings(scene)}/{len(scene.mdt_group_mappings)}",
            icon="LINKED",
        )
        if len(scene.mdt_group_mappings) == 0:
            mapping_box.label(text="No mappings stored.", icon="INFO")
        else:
            mapping_box.template_list(
                "MDT_UL_group_mappings",
                "",
                scene,
                "mdt_group_mappings",
                scene,
                "mdt_active_group_mapping_index",
                rows=min(7, max(4, len(scene.mdt_group_mappings))),
            )

        apply_box = layout.box()
        apply_col = apply_box.column(align=True)
        apply_col.prop(settings, "apply_source_object")
        apply_col.prop(settings, "apply_target_object")
        apply_col.prop(settings, "invert_mapping")
        apply_row = apply_box.row(align=True)
        apply_row.enabled = (
            apply_source is not None
            and apply_target is not None
            and apply_source != apply_target
            and count_enabled_group_mappings(scene) > 0
        )
        apply_row.operator("mdt.apply_seam_mapping", icon="MOD_VERTEX_WEIGHT")


class MDT_PT_materials(_MDT_PT_base, bpy.types.Panel):
    """Sidebar panel for material creation and material-based splitting."""

    bl_label = "Materials"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, _context):
        return True

    def draw(self, context):
        layout = self.layout
        has_mesh_selection = len(get_selected_mesh_objects(context)) > 0

        create_row = layout.row()
        create_row.enabled = has_mesh_selection
        create_row.operator("mdt.create_named_materials", icon="MATERIAL")

        split_row = layout.row()
        split_row.enabled = has_mesh_selection
        split_row.operator("mdt.separate_by_material", icon="MOD_EXPLODE")

        if not has_mesh_selection:
            layout.label(text="Select one or more mesh objects.", icon="INFO")
