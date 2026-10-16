"""Property definitions for Mod Tool."""

import bpy


def _mesh_object_poll(_self, obj):
    """Restrict object pickers to mesh objects."""
    return obj is not None and getattr(obj, "type", None) == "MESH"


def _armature_object_poll(_self, obj):
    """Restrict object pickers to armature objects."""
    return obj is not None and getattr(obj, "type", None) == "ARMATURE"


def _pose_mirror_enabled_update(_self, _context):
    from .core.pose_mirror import clear_pose_mirror_state

    clear_pose_mirror_state()


def _rig_mapping_items(self, context):
    from .core.rig_group_rename import rig_mapping_items

    return rig_mapping_items(self, context)


class MDT_PG_AnimationSettings(bpy.types.PropertyGroup):
    """File and destination for standard MMD body retargeting."""

    target_armature: bpy.props.PointerProperty(
        name="Target Rig", type=bpy.types.Object, poll=_armature_object_poll,
    )
    translation_multiplier: bpy.props.FloatProperty(
        name="Translation Multiplier", description="Multiplier for motion translations after automatic leg-length scaling; 1 uses the Tda reference proportions",
        default=1., min=.001, soft_max=2, precision=3,
    )
    start_frame: bpy.props.IntProperty(name="Start Frame", default=1, min=0)
    last_action: bpy.props.StringProperty()
    last_report: bpy.props.StringProperty()


class MDT_PG_PoseMirrorSettings(bpy.types.PropertyGroup):
    """Settings for lightweight automatic pose mirroring."""

    enabled: bpy.props.BoolProperty(
        name="Smart Pose Mirror",
        description="Mirror selected pose-bone transforms to the opposite side while editing in Pose Mode",
        default=False,
        update=_pose_mirror_enabled_update,
    )


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

    target_armature: bpy.props.PointerProperty(
        name="Target Rig",
        description="Armature whose actual bone names determine the rename targets",
        type=bpy.types.Object,
        poll=_armature_object_poll,
    )

    rig_mapping: bpy.props.EnumProperty(
        name="Mapping",
        description="Mapping preset used for the active mesh vertex groups",
        items=_rig_mapping_items,
    )


class MDT_PG_BaseColorSPSettings(bpy.types.PropertyGroup):
    """Settings for the reversible single-material Substance Painter workflow."""

    grid_size: bpy.props.EnumProperty(
        name="Grid",
        description="Fixed atlas grid used for Base Color material preparation",
        items=(
            ("2", "2 x 2", "Place up to four materials in a 2 by 2 grid"),
            ("3", "3 x 3", "Place up to nine materials in a 3 by 3 grid"),
            ("4", "4 x 4", "Place up to sixteen materials in a 4 by 4 grid"),
        ),
        default="2",
    )

    directory: bpy.props.StringProperty(
        name="Texture Directory",
        description="Directory for the SP Base Color atlas and restored material textures",
        subtype="DIR_PATH",
        default="//",
    )


class MDT_PG_ColorBakeMaterial(bpy.types.PropertyGroup):
    slot_index: bpy.props.IntProperty(default=0, min=0)
    material: bpy.props.PointerProperty(type=bpy.types.Material)
    target_uv: bpy.props.StringProperty(name="Destination UV")


class MDT_PG_ColorBakeSettings(bpy.types.PropertyGroup):
    mesh_object: bpy.props.PointerProperty(type=bpy.types.Object)
    source_uv: bpy.props.StringProperty(name="Source UV", description="Captured active UV for texture nodes without an explicit UV map")
    materials: bpy.props.CollectionProperty(type=MDT_PG_ColorBakeMaterial)
    filepath: bpy.props.StringProperty(name="Output PNG", subtype="FILE_PATH", default="//OutputColor.png")
    resolution: bpy.props.EnumProperty(name="Resolution", items=(("1024", "1K", "1024 pixels"), ("2048", "2K", "2048 pixels"), ("4096", "4K", "4096 pixels"), ("8192", "8K", "8192 pixels")), default="4096")
    margin: bpy.props.IntProperty(name="Margin", description="Padding in pixels", default=8, min=0, max=64)
    last_image: bpy.props.PointerProperty(type=bpy.types.Image)
    last_uv: bpy.props.StringProperty()


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


class MDT_PG_JointSculptSelection(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(type=bpy.types.Object)


class MDT_PG_JointSculptSource(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(type=bpy.types.Object)
    mesh: bpy.props.PointerProperty(type=bpy.types.Mesh)
    mesh_users: bpy.props.IntProperty()
    start: bpy.props.IntProperty()
    count: bpy.props.IntProperty()
    hidden: bpy.props.BoolProperty()
    state_hash: bpy.props.StringProperty()


class MDT_PG_JointSculptSettings(bpy.types.PropertyGroup):
    proxy: bpy.props.PointerProperty(type=bpy.types.Object)
    proxy_mesh: bpy.props.PointerProperty(type=bpy.types.Mesh)
    sources: bpy.props.CollectionProperty(type=MDT_PG_JointSculptSource)
    selection: bpy.props.CollectionProperty(type=MDT_PG_JointSculptSelection)
    previous_active: bpy.props.PointerProperty(type=bpy.types.Object)
    previous_mode: bpy.props.StringProperty()
    view_layer_name: bpy.props.StringProperty()
    topology_hash: bpy.props.StringProperty()
    proxy_matrix: bpy.props.FloatVectorProperty(size=16)
    brush: bpy.props.PointerProperty(type=bpy.types.Brush)
    automasking_backup: bpy.props.StringProperty()


REGISTERED_CLASSES = (
    MDT_PG_AnimationSettings,
    MDT_PG_PoseMirrorSettings,
    MDT_PG_RenameSettings,
    MDT_PG_GroupMatchSettings,
    MDT_PG_BaseColorSPSettings,
    MDT_PG_ColorBakeMaterial,
    MDT_PG_ColorBakeSettings,
    MDT_PG_GroupMatchResultItem,
    MDT_PG_ShapeKeyTransferSettings,
    MDT_PG_BoneMergeSettings,
    MDT_PG_JointSculptSelection,
    MDT_PG_JointSculptSource,
    MDT_PG_JointSculptSettings,
)


def register_addon_properties():
    """Register addon PointerProperties and collections on Scene."""
    bpy.types.Scene.mdt_animation_settings = bpy.props.PointerProperty(type=MDT_PG_AnimationSettings)
    bpy.types.Scene.mdt_pose_mirror_settings = bpy.props.PointerProperty(type=MDT_PG_PoseMirrorSettings)
    bpy.types.Scene.mdt_rename_settings = bpy.props.PointerProperty(type=MDT_PG_RenameSettings)
    bpy.types.Scene.mdt_group_match_settings = bpy.props.PointerProperty(type=MDT_PG_GroupMatchSettings)
    bpy.types.Scene.mdt_base_color_sp_settings = bpy.props.PointerProperty(type=MDT_PG_BaseColorSPSettings)
    bpy.types.Scene.mdt_color_bake_settings = bpy.props.PointerProperty(type=MDT_PG_ColorBakeSettings)
    bpy.types.Scene.mdt_joint_sculpt_settings = bpy.props.PointerProperty(type=MDT_PG_JointSculptSettings)
    bpy.types.Scene.mdt_shape_key_settings = bpy.props.PointerProperty(type=MDT_PG_ShapeKeyTransferSettings)
    bpy.types.Scene.mdt_bone_merge_settings = bpy.props.PointerProperty(type=MDT_PG_BoneMergeSettings)
    bpy.types.Scene.mdt_group_match_results = bpy.props.CollectionProperty(type=MDT_PG_GroupMatchResultItem)


def unregister_addon_properties():
    """Remove addon PointerProperties and collections from Scene."""
    if hasattr(bpy.types.Scene, "mdt_animation_settings"):
        del bpy.types.Scene.mdt_animation_settings
    if hasattr(bpy.types.Scene, "mdt_joint_sculpt_settings"):
        del bpy.types.Scene.mdt_joint_sculpt_settings
    if hasattr(bpy.types.Scene, "mdt_color_bake_settings"):
        del bpy.types.Scene.mdt_color_bake_settings
    if hasattr(bpy.types.Scene, "mdt_pose_mirror_settings"):
        del bpy.types.Scene.mdt_pose_mirror_settings
    if hasattr(bpy.types.Scene, "mdt_group_match_results"):
        del bpy.types.Scene.mdt_group_match_results
    if hasattr(bpy.types.Scene, "mdt_shape_key_settings"):
        del bpy.types.Scene.mdt_shape_key_settings
    if hasattr(bpy.types.Scene, "mdt_bone_merge_settings"):
        del bpy.types.Scene.mdt_bone_merge_settings
    if hasattr(bpy.types.Scene, "mdt_group_match_settings"):
        del bpy.types.Scene.mdt_group_match_settings
    if hasattr(bpy.types.Scene, "mdt_base_color_sp_settings"):
        del bpy.types.Scene.mdt_base_color_sp_settings
    if hasattr(bpy.types.Scene, "mdt_rename_settings"):
        del bpy.types.Scene.mdt_rename_settings
