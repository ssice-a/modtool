"""Property definitions for Mod Tool."""

import bpy


def _mesh_object_poll(_self, obj):
    """Restrict object pickers to mesh objects."""
    return obj is not None and getattr(obj, "type", None) == "MESH"


def _armature_object_poll(_self, obj):
    """Restrict object pickers to armature objects."""
    return obj is not None and getattr(obj, "type", None) == "ARMATURE"


class MDT_PG_RenameSettings(bpy.types.PropertyGroup):
    """Settings used by the vertex-group rename tool."""

    mode: bpy.props.EnumProperty(
        name="Mode",
        description="Choose how the selected vertex groups should be renamed",
        items=(
            ("MODIFY", "Rules", "Apply search, replace, prefix, and suffix rules"),
            ("SEQUENCE", "Sequence", "Rename all vertex groups as a numbered series"),
        ),
        default="MODIFY",
    )

    search: bpy.props.StringProperty(
        name="Search",
        description="Text to search for. Leave empty to only apply prefix and suffix",
        default="",
    )

    replace: bpy.props.StringProperty(
        name="Replace",
        description="Replacement text. In regex mode this follows Python regular-expression replacement syntax",
        default="",
    )

    use_regex: bpy.props.BoolProperty(
        name="Regex",
        description="Treat the search text as a regular expression",
        default=False,
    )

    case_sensitive: bpy.props.BoolProperty(
        name="Match Case",
        description="When disabled, search and replace ignores letter case",
        default=False,
    )

    prefix: bpy.props.StringProperty(
        name="Prefix",
        description="Text added before every vertex-group name",
        default="",
    )

    suffix: bpy.props.StringProperty(
        name="Suffix",
        description="Text added after every vertex-group name",
        default="",
    )

    sequence_base: bpy.props.StringProperty(
        name="Base Name",
        description="Base name used by sequential renaming mode",
        default="Group",
    )

    sequence_separator: bpy.props.StringProperty(
        name="Separator",
        description="Text inserted between the base name and the number",
        default="_",
    )

    sequence_start: bpy.props.IntProperty(
        name="Start Index",
        description="Starting number used by sequential renaming mode",
        default=1,
        min=0,
    )

    sequence_padding: bpy.props.IntProperty(
        name="Zero Padding",
        description="Number width used by sequential renaming mode",
        default=2,
        min=1,
        max=8,
    )


class MDT_PG_GroupMatchSettings(bpy.types.PropertyGroup):
    """Settings used for source-to-target vertex-group mapping."""

    source_object: bpy.props.PointerProperty(
        name="Source",
        description="Mesh object whose deform-bone vertex groups will be renamed",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
    )

    target_object: bpy.props.PointerProperty(
        name="Target",
        description="Mesh object whose vertex-group names will be used as rename targets",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
    )


class MDT_PG_GroupMatchResultItem(bpy.types.PropertyGroup):
    """One source-to-target rename row inferred from spatial weight matching."""

    source_group: bpy.props.StringProperty(
        name="Source Group",
        description="Source vertex-group name that may be renamed",
        default="",
    )

    target_group: bpy.props.StringProperty(
        name="Suggested Target",
        description="Suggested target vertex-group name",
        default="",
    )

    status: bpy.props.EnumProperty(
        name="Status",
        description="Mapping confidence status",
        items=(
            ("DIRECT", "Direct", "A one-to-one spatial rename match"),
            ("MERGE", "Merge", "Multiple source groups should be merged into one target group"),
            ("FALLBACK", "Fallback", "A low-confidence mapping used to keep every source bone group mapped"),
        ),
        default="FALLBACK",
    )

    confidence: bpy.props.FloatProperty(
        name="Confidence",
        description="Final confidence score for the suggested target group",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=4,
    )

    score_overlap: bpy.props.FloatProperty(
        name="Cloud",
        description="Spatial weight-cloud vote score between source and target groups",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=4,
    )

    score_centroid: bpy.props.FloatProperty(
        name="Centroid",
        description="Centroid similarity score between source and target groups",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=4,
    )

    reason: bpy.props.StringProperty(
        name="Reason",
        description="Short explanation for the current mapping status",
        default="",
    )


class MDT_PG_ShapeKeyTransferSettings(bpy.types.PropertyGroup):
    """Settings used for cross-mesh shape-key transfer."""

    transfer_mode: bpy.props.EnumProperty(
        name="Mode",
        description="Choose how shape keys are matched between the source and target meshes",
        items=(
            (
                "WRAP",
                "Wrap Bake",
                "Bind the target to the source with Surface Deform and bake each source key from its reference pose to its full pose",
            ),
            (
                "VERTEX",
                "Vertex",
                "Use nearby source vertices as the main match and bias them with corresponding face-corner hints",
            ),
            ("SURFACE", "Surface", "Use nearest points on the source surface to match the target mesh"),
        ),
        default="WRAP",
    )

    source_object: bpy.props.PointerProperty(
        name="Source",
        description="Mesh object that already has the source shape keys",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
    )

    target_object: bpy.props.PointerProperty(
        name="Target",
        description="Mesh object that will receive the transferred shape keys",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
    )

    max_distance: bpy.props.FloatProperty(
        name="Surface Tolerance",
        description="Maximum world-space distance allowed when snapping each target vertex to the source surface",
        default=0.001,
        min=0.0,
        soft_min=0.000001,
        soft_max=0.1,
        precision=6,
    )

    vertex_tolerance: bpy.props.FloatProperty(
        name="Vertex Tolerance",
        description="Maximum world-space distance used for the nearby source-vertex candidates before face hints add extra guidance",
        default=0.0001,
        min=0.0,
        soft_min=0.000001,
        soft_max=0.01,
        precision=6,
    )

    shared_vertex_tolerance: bpy.props.FloatProperty(
        name="Shared Vertex Tolerance",
        description="World-space distance used to weld overlapping source and target vertices into shared groups; 0 means exact overlap only",
        default=0.0,
        min=0.0,
        soft_min=0.0,
        soft_max=0.001,
        precision=6,
    )

    replace_existing: bpy.props.BoolProperty(
        name="Replace Same Name",
        description="Replace target shape keys that already use the same name as the source keys",
        default=True,
    )


class MDT_PG_BoneMergeSettings(bpy.types.PropertyGroup):
    """Settings used to transfer selected source rest bones into a target armature."""

    source_object: bpy.props.PointerProperty(
        name="Source",
        description="Armature whose selected bones provide the body rest shape",
        type=bpy.types.Object,
        poll=_armature_object_poll,
    )

    target_object: bpy.props.PointerProperty(
        name="Target",
        description="Armature that will receive the selected source bones",
        type=bpy.types.Object,
        poll=_armature_object_poll,
    )

    captured_bone_names: bpy.props.StringProperty(
        name="Captured Bones",
        description="Internal newline-separated list of selected source bone names",
        default="",
    )

    use_source_parenting: bpy.props.BoolProperty(
        name="Use Source Parents",
        description="Reconnect transferred bones using the source bone parent chain when possible",
        default=True,
    )


REGISTERED_CLASSES = (
    MDT_PG_RenameSettings,
    MDT_PG_GroupMatchSettings,
    MDT_PG_GroupMatchResultItem,
    MDT_PG_ShapeKeyTransferSettings,
    MDT_PG_BoneMergeSettings,
)


def register_addon_properties():
    """Register addon PointerProperties and collections on Scene."""
    bpy.types.Scene.mdt_rename_settings = bpy.props.PointerProperty(type=MDT_PG_RenameSettings)
    bpy.types.Scene.mdt_group_match_settings = bpy.props.PointerProperty(type=MDT_PG_GroupMatchSettings)
    bpy.types.Scene.mdt_shape_key_settings = bpy.props.PointerProperty(type=MDT_PG_ShapeKeyTransferSettings)
    bpy.types.Scene.mdt_bone_merge_settings = bpy.props.PointerProperty(type=MDT_PG_BoneMergeSettings)
    bpy.types.Scene.mdt_group_match_results = bpy.props.CollectionProperty(type=MDT_PG_GroupMatchResultItem)


def unregister_addon_properties():
    """Remove addon PointerProperties and collections from Scene."""
    if hasattr(bpy.types.Scene, "mdt_group_match_results"):
        del bpy.types.Scene.mdt_group_match_results
    if hasattr(bpy.types.Scene, "mdt_shape_key_settings"):
        del bpy.types.Scene.mdt_shape_key_settings
    if hasattr(bpy.types.Scene, "mdt_bone_merge_settings"):
        del bpy.types.Scene.mdt_bone_merge_settings
    if hasattr(bpy.types.Scene, "mdt_group_match_settings"):
        del bpy.types.Scene.mdt_group_match_settings
    if hasattr(bpy.types.Scene, "mdt_rename_settings"):
        del bpy.types.Scene.mdt_rename_settings
