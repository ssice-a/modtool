"""UI registration for Mod Tool."""

from .panels import (
    MDT_PT_animation,
    MDT_PT_armature_cleanup,
    MDT_PT_materials,
    MDT_PT_mesh,
    MDT_PT_proxy_fit,
    MDT_PT_shape_keys,
    MDT_PT_vertex_groups,
)


REGISTERED_CLASSES = (
    MDT_PT_animation,
    MDT_PT_mesh,
    MDT_PT_proxy_fit,
    MDT_PT_vertex_groups,
    MDT_PT_shape_keys,
    MDT_PT_materials,
    MDT_PT_armature_cleanup,
)
