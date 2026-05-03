"""UI registration for Mod Tool."""

from .lists import MDT_UL_group_match_results
from .panels import (
    MDT_PT_armature_cleanup,
    MDT_PT_materials,
    MDT_PT_shape_keys,
    MDT_PT_vertex_groups,
)


REGISTERED_CLASSES = (
    MDT_UL_group_match_results,
    MDT_PT_vertex_groups,
    MDT_PT_shape_keys,
    MDT_PT_materials,
    MDT_PT_armature_cleanup,
)
