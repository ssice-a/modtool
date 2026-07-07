"""Operator registration for Mod Tool."""

from .armature_cleanup import MDT_OT_clean_fk_armature
from .bone_merge import (
    MDT_OT_analyze_bone_merge,
    MDT_OT_apply_bone_merge,
    MDT_OT_capture_bone_merge_selection,
)
from .group_mapping import MDT_OT_rename_groups_from_target, MDT_OT_unify_seam_weights_from_target
from .materials import MDT_OT_create_named_materials, MDT_OT_separate_by_material
from .modifier_apply import (
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_normalize_shape_key_ranges,
    MDT_OT_set_current_shape_key_values_as_zero,
)
from .rename import MDT_OT_batch_rename_groups, MDT_OT_keep_body_vertex_groups
from .rename import MDT_OT_clean_disconnected_vertex_group_weights
from .rename import MDT_OT_keep_deforming_vertex_groups
from .rename import MDT_OT_mirror_vertex_group_weights
from .rename import MDT_OT_remove_empty_vertex_groups, MDT_OT_reset_rename_settings
from .seam import MDT_OT_rename_selected_seam_groups
from .shape_keys import MDT_OT_transfer_shape_keys


REGISTERED_CLASSES = (
    MDT_OT_batch_rename_groups,
    MDT_OT_remove_empty_vertex_groups,
    MDT_OT_clean_disconnected_vertex_group_weights,
    MDT_OT_keep_deforming_vertex_groups,
    MDT_OT_keep_body_vertex_groups,
    MDT_OT_mirror_vertex_group_weights,
    MDT_OT_reset_rename_settings,
    MDT_OT_rename_groups_from_target,
    MDT_OT_unify_seam_weights_from_target,
    MDT_OT_create_named_materials,
    MDT_OT_separate_by_material,
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_set_current_shape_key_values_as_zero,
    MDT_OT_normalize_shape_key_ranges,
    MDT_OT_rename_selected_seam_groups,
    MDT_OT_transfer_shape_keys,
    MDT_OT_clean_fk_armature,
    MDT_OT_capture_bone_merge_selection,
    MDT_OT_analyze_bone_merge,
    MDT_OT_apply_bone_merge,
)
