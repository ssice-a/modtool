"""Mod Tool addon entry point."""

bl_info = {
    "name": "Mod Tool",
    "author": "OpenAI Codex",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Mod Tool",
    "description": "Mesh modding helpers for shape keys, vertex groups, seam renaming, and material splits.",
    "category": "Mesh",
}

import bpy

from . import properties
from .ops import REGISTERED_CLASSES as OPERATOR_CLASSES
from .ui import REGISTERED_CLASSES as UI_CLASSES


REGISTERED_CLASSES = properties.REGISTERED_CLASSES + OPERATOR_CLASSES + UI_CLASSES


def register():
    """Register addon classes and properties."""
    for blender_class in REGISTERED_CLASSES:
        bpy.utils.register_class(blender_class)
    properties.register_addon_properties()


def unregister():
    """Unregister addon classes and properties."""
    properties.unregister_addon_properties()
    for blender_class in reversed(REGISTERED_CLASSES):
        bpy.utils.unregister_class(blender_class)


if __name__ == "__main__":
    register()
