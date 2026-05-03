"""Operator registration for Mod Tool."""

from .armature_cleanup import MDT_OT_clean_fk_armature
from .group_mapping import MDT_OT_analyze_group_mapping, MDT_OT_apply_group_mapping_rename
from .group_mapping import MDT_OT_accept_group_match_candidates, MDT_OT_add_group_match_row, MDT_OT_remove_group_match_row
from .materials import MDT_OT_create_named_materials, MDT_OT_separate_by_material
from .modifier_apply import MDT_OT_apply_modifier_keep_shape_keys
from .modifier_apply import MDT_OT_delete_selected_geometry_keep_shape_keys
from .rename import MDT_OT_batch_rename_groups, MDT_OT_remove_empty_vertex_groups, MDT_OT_reset_rename_settings
from .seam import MDT_OT_rename_selected_seam_groups
from .shape_keys import MDT_OT_transfer_shape_keys


REGISTERED_CLASSES = (
    MDT_OT_batch_rename_groups,
    MDT_OT_remove_empty_vertex_groups,
    MDT_OT_reset_rename_settings,
    MDT_OT_analyze_group_mapping,
    MDT_OT_apply_group_mapping_rename,
    MDT_OT_accept_group_match_candidates,
    MDT_OT_add_group_match_row,
    MDT_OT_remove_group_match_row,
    MDT_OT_create_named_materials,
    MDT_OT_separate_by_material,
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_delete_selected_geometry_keep_shape_keys,
    MDT_OT_rename_selected_seam_groups,
    MDT_OT_transfer_shape_keys,
    MDT_OT_clean_fk_armature,
)
