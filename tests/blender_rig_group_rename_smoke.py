"""Blender background regression test for dynamic game-rig group renaming."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


REPOSITORY_PARENT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_PARENT not in sys.path:
    sys.path.insert(0, REPOSITORY_PARENT)

from modtool.core.rig_group_rename import (  # noqa: E402
    apply_auto_rig_npc,
    apply_auto_rig_npc_batch,
)


def _reset_scene():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _create_armature(name, bone_names):
    armature_data = bpy.data.armatures.new(f"{name}Data")
    armature = bpy.data.objects.new(name, armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for index, bone_name in enumerate(bone_names):
        bone = armature_data.edit_bones.new(bone_name)
        bone.head = (0.0, 0.0, float(index))
        bone.tail = (0.0, 0.0, float(index + 1))
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _create_target_mesh(name, group_names):
    mesh = bpy.data.meshes.new(f"{name}Data")
    mesh.from_pydata(((0.0, 0.0, 0.0),), (), ())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for group_name in group_names:
        obj.vertex_groups.new(name=group_name)
    return obj


def _weight(obj, group_name, vertex_index):
    try:
        return obj.vertex_groups[group_name].weight(vertex_index)
    except RuntimeError:
        return 0.0


def _test_dynamic_chest_variant_preserves_armature_modifier():
    _reset_scene()
    _create_target_mesh(
        "TargetMesh",
        (
            "Bip001-Pelvis",
            "Bn_l_breast_001",
            "Bn_r_breast_001",
            "Bn_l_breast_002",
            "Bn_r_breast_002",
            "Bn_l_butt_001",
            "Bn_r_butt_001",
            "Bip001-L-Finger1",
            "Bip001-L-Finger11",
            "Bip001-L-Finger12",
        ),
    )
    source_rig = _create_armature("SourceRig", ("root.x",))

    mesh = bpy.data.meshes.new("SourceMesh")
    mesh.from_pydata(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (), ())
    source = bpy.data.objects.new("SourceMesh", mesh)
    bpy.context.collection.objects.link(source)
    modifier = source.modifiers.new("OriginalArmature", "ARMATURE")
    modifier.object = source_rig

    source.vertex_groups.new(name="root.x").add((0, 1), 0.5, "REPLACE")
    source.vertex_groups.new(name="Butt.L").add((0,), 0.5, "REPLACE")
    source.vertex_groups.new(name="Breast.L").add((1,), 0.5, "REPLACE")
    source.vertex_groups.new(name="c_index1_base.l").add((1,), 0.25, "REPLACE")
    source.vertex_groups.new(name="index1.l").add((1,), 0.25, "REPLACE")
    source.vertex_groups.new(name="c_index2.l").add((1,), 0.25, "REPLACE")
    source.vertex_groups.new(name="c_index3.l").add((1,), 0.25, "REPLACE")
    source.vertex_groups.new(name="Mask").add((0, 1), 1.0, "REPLACE")
    modifier_before = tuple((item.name, item.type, item.object) for item in source.modifiers)

    result = apply_auto_rig_npc(source)

    assert {group.name for group in source.vertex_groups} == {
        "Bip001-Pelvis",
        "Bn_r_breast_002",
        "Bn_l_butt_001",
        "Bip001-L-Finger1",
        "Bip001-L-Finger11",
        "Bip001-L-Finger12",
    }
    assert abs(_weight(source, "Bip001-Pelvis", 0) - 0.5) < 1.0e-6
    assert abs(_weight(source, "Bn_l_butt_001", 0) - 0.5) < 1.0e-6
    assert abs(_weight(source, "Bip001-Pelvis", 1) - 0.25) < 1.0e-6
    assert abs(_weight(source, "Bn_r_breast_002", 1) - 0.25) < 1.0e-6
    assert abs(_weight(source, "Bip001-L-Finger1", 1) - 0.125) < 1.0e-6
    assert abs(_weight(source, "Bip001-L-Finger11", 1) - 0.25) < 1.0e-6
    assert abs(_weight(source, "Bip001-L-Finger12", 1) - 0.125) < 1.0e-6
    assert tuple((item.name, item.type, item.object) for item in source.modifiers) == modifier_before
    assert result["removed_group_count"] == 1
    print("DYNAMIC_GAME_RIG_RENAME_OK")


def _test_breast_alias_merge_and_unmatched_groups_are_removed():
    _reset_scene()
    _create_target_mesh(
        "TargetBreastMesh",
        (
            "Bn_l_breast_002",
            "Bn_r_breast_002",
        ),
    )

    mesh = bpy.data.meshes.new("BreastSourceMesh")
    mesh.from_pydata(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (), ())
    source = bpy.data.objects.new("BreastSourceMesh", mesh)
    bpy.context.collection.objects.link(source)
    source.vertex_groups.new(name="Breast.L").add((0,), 0.4, "REPLACE")
    source.vertex_groups.new(name="Breast.L.001").add((0,), 0.6, "REPLACE")
    source.vertex_groups.new(name="Breast.R").add((1,), 0.25, "REPLACE")
    source.vertex_groups.new(name="Breast.R.001").add((1,), 0.75, "REPLACE")
    source.vertex_groups.new(name="B_Line").add((0, 1), 1.0, "REPLACE")
    source.vertex_groups.new(name="mask_Arm").add((0, 1), 1.0, "REPLACE")

    result = apply_auto_rig_npc(source)

    assert {group.name for group in source.vertex_groups} == {
        "Bn_l_breast_002",
        "Bn_r_breast_002",
    }
    assert abs(_weight(source, "Bn_l_breast_002", 1) - 1.0) < 1.0e-6
    assert abs(_weight(source, "Bn_r_breast_002", 0) - 1.0) < 1.0e-6
    assert result["mapped_source_count"] == 4
    assert result["removed_group_count"] == 2
    print("BREAST_ALIAS_MERGE_OK")


def _test_batch_rename_requires_no_armature():
    _reset_scene()
    sources = []
    for index in range(2):
        mesh = bpy.data.meshes.new(f"BatchSourceMesh{index}Data")
        mesh.from_pydata(((0.0, 0.0, 0.0),), (), ())
        source = bpy.data.objects.new(f"BatchSourceMesh{index}", mesh)
        bpy.context.collection.objects.link(source)
        source.vertex_groups.new(name="root.x").add((0,), 1.0, "REPLACE")
        source.vertex_groups.new(name="Mask").add((0,), 1.0, "REPLACE")
        sources.append(source)

    results = apply_auto_rig_npc_batch(sources)

    assert len(results) == 2
    for source, result in results:
        assert {group.name for group in source.vertex_groups} == {"Bip001-Pelvis"}
        assert abs(_weight(source, "Bip001-Pelvis", 0) - 1.0) < 1.0e-6
        assert result["removed_group_count"] == 1
    print("BATCH_WITHOUT_ARMATURE_OK")


if __name__ == "__main__":
    _test_dynamic_chest_variant_preserves_armature_modifier()
    _test_breast_alias_merge_and_unmatched_groups_are_removed()
    _test_batch_rename_requires_no_armature()
    print("RIG_GROUP_RENAME_SMOKE_OK")
