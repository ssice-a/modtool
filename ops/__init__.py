"""Operator registration for Mod Tool."""

from .armature_cleanup import MDT_OT_clean_fk_armature
from .group_mapping import MDT_OT_rename_groups_from_target
from .materials import MDT_OT_create_named_materials, MDT_OT_separate_by_material
from .modifier_apply import MDT_OT_apply_modifier_keep_shape_keys
from .modifier_apply import MDT_OT_delete_selected_geometry_keep_shape_keys
from .rename import MDT_OT_batch_rename_groups, MDT_OT_keep_deforming_vertex_groups
from .rename import MDT_OT_remove_empty_vertex_groups, MDT_OT_reset_rename_settings
from .seam import MDT_OT_rename_selected_seam_groups
from .shape_keys import MDT_OT_transfer_shape_keys


REGISTERED_CLASSES = (
    MDT_OT_batch_rename_groups,
    MDT_OT_remove_empty_vertex_groups,
    MDT_OT_keep_deforming_vertex_groups,
    MDT_OT_reset_rename_settings,
    MDT_OT_rename_groups_from_target,
    MDT_OT_create_named_materials,
    MDT_OT_separate_by_material,
    MDT_OT_apply_modifier_keep_shape_keys,
    MDT_OT_delete_selected_geometry_keep_shape_keys,
    MDT_OT_rename_selected_seam_groups,
    MDT_OT_transfer_shape_keys,
    MDT_OT_clean_fk_armature,
)
