"""Operator registration for Mod Tool."""

from .armature_cleanup import MDT_OT_clean_fk_armature
from .mmd_animation import MDT_OT_import_mmd_animation
from .bone_merge import (
    MDT_OT_analyze_bone_merge,
    MDT_OT_apply_bone_merge,
    MDT_OT_capture_bone_merge_selection,
)
from .bridge_loops import MDT_OT_bridge_uv_loops
from .group_mapping import MDT_OT_rename_groups_from_target, MDT_OT_unify_seam_weights_from_target
from .rig_group_rename import MDT_OT_auto_rig_npc
from .materials import (
    MDT_OT_create_named_materials,
    MDT_OT_merge_base_color_for_sp,
    MDT_OT_restore_base_color_from_sp,
    MDT_OT_separate_by_material,
)
from .material_bake import MDT_OT_prepare_color_bake, MDT_OT_bake_output_colors
from .joint_sculpt import MDT_OT_start_joint_sculpt, MDT_OT_apply_joint_sculpt, MDT_OT_cancel_joint_sculpt
from .mesh_normals import MDT_OT_merge_selected_vertex_normals_active, MDT_OT_merge_selected_vertex_normals_average
from .modifier_apply import (
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_apply_unlocked_shape_keys,
)
from .rename import MDT_OT_batch_rename_groups, MDT_OT_keep_body_vertex_groups
from .rename import MDT_OT_clean_disconnected_vertex_group_weights
from .rename import MDT_OT_keep_deforming_vertex_groups
from .rename import MDT_OT_mirror_vertex_group_weights
from .rename import MDT_OT_remove_empty_vertex_groups, MDT_OT_reset_rename_settings
from .seam import MDT_OT_rename_selected_seam_groups
from .shape_keys import MDT_OT_transfer_shape_keys


REGISTERED_CLASSES = (
    MDT_OT_import_mmd_animation,
    MDT_OT_bridge_uv_loops,
    MDT_OT_start_joint_sculpt,
    MDT_OT_apply_joint_sculpt,
    MDT_OT_cancel_joint_sculpt,
    MDT_OT_batch_rename_groups,
    MDT_OT_remove_empty_vertex_groups,
    MDT_OT_clean_disconnected_vertex_group_weights,
    MDT_OT_keep_deforming_vertex_groups,
    MDT_OT_keep_body_vertex_groups,
    MDT_OT_mirror_vertex_group_weights,
    MDT_OT_reset_rename_settings,
    MDT_OT_rename_groups_from_target,
    MDT_OT_unify_seam_weights_from_target,
    MDT_OT_auto_rig_npc,
    MDT_OT_merge_selected_vertex_normals_average,
    MDT_OT_merge_selected_vertex_normals_active,
    MDT_OT_create_named_materials,
    MDT_OT_merge_base_color_for_sp,
    MDT_OT_restore_base_color_from_sp,
    MDT_OT_prepare_color_bake,
    MDT_OT_bake_output_colors,
    MDT_OT_separate_by_material,
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_apply_unlocked_shape_keys,
    MDT_OT_rename_selected_seam_groups,
    MDT_OT_transfer_shape_keys,
    MDT_OT_clean_fk_armature,
    MDT_OT_capture_bone_merge_selection,
    MDT_OT_analyze_bone_merge,
    MDT_OT_apply_bone_merge,
)
