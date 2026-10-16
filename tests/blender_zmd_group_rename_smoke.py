"""Focused ZMD preset and selected-mesh operator regression tests."""

import sys
from pathlib import Path

import bpy
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modtool.core.rig_group_rename import (
    AUTO_RIG_ZMD_MAPPING_ID,
    VRC_ZMD_MAPPING_ID,
    _build_mapping,
    apply_auto_rig_npc,
    rig_mapping_items,
)
from modtool.ops.rig_group_rename import MDT_OT_auto_rig_npc
from modtool.properties import MDT_PG_GroupMatchSettings


def weights(obj):
    names = {g.index: g.name for g in obj.vertex_groups}
    return [{names[a.group]: a.weight for a in v.groups} for v in obj.data.vertices]


def dense(collection, prop, width):
    values = np.empty(len(collection) * width, dtype=np.float32)
    collection.foreach_get(prop, values)
    return values


def mesh_snapshot(obj):
    mesh = obj.data
    return (
        dense(mesh.vertices, "co", 3).tobytes(),
        dense(mesh.loops, "vertex_index", 1).tobytes(),
        dense(mesh.uv_layers.active.data, "uv", 2).tobytes(),
        tuple((k.name, k.value, dense(k.data, "co", 3).tobytes())
              for k in mesh.shape_keys.key_blocks),
        obj.parent,
        tuple((m.name, m.type, m.object) for m in obj.modifiers),
    )


def fixture(name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i % 4, i // 4, 0) for i in range(12)], [],
                     [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    uv = mesh.uv_layers.new(name="OriginalUV")
    uv.data.foreach_set("uv", np.arange(len(mesh.loops) * 2, dtype=np.float32) / 10)
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Sculpt")
    key.data.foreach_set("co", dense(mesh.vertices, "co", 3) + 0.2)
    key.value = 0.65
    assignments = [
        {"Breast.R": 0.3, "Breast.R.001": 0.2, "Breast_0.R": 0.1, "root.x": 0.4},
        {"Breast.L": 0.1, "Breast.L.001": 0.2, "Breast_0.L": 0.2, "root.x": 0.5},
        {"arm_stretch.l": 0.7, "c_arm_twist_offset.l": 0.3},
        {"forearm_stretch.r": 0.4, "forearm_twist.r": 0.6},
        {"BigToe.L": 0.2, "SmallToe4_2.L": 0.3, "foot.l": 0.5},
        {"Butt.R": 0.2, "Vagina_bone_1": 0.5, "root.x": 0.3},
        {"index1.l": 0.2, "c_index2.l": 0.3, "c_index3.l": 0.5},
        {"Bip001_L_UpperArm_corrective_ty_plus": 0.2, "root.x": 0.8},
        {"Breast.R": 0.3, "Breast.R.001": 0.2, "Breast_0.R": 0.1},
        {"Breast.L": 0.3, "Breast.L.001": 0.2, "Breast_0.L": 0.1},
        # Real Body pattern: missing breasts used to amplify the tiny
        # surviving clavicle influence to 1.0 during normalization.
        {"Breast.R": 0.590825, "Breast.R.001": 0.330055, "shoulder.l": 6.31e-8},
        {"Breast.L": 0.590825, "Breast.L.001": 0.330055, "shoulder.r": 6.31e-8},
    ]
    for index, row in enumerate(assignments):
        for name, value in row.items():
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([index], value, "REPLACE")
    for name in ("B_Line", "Body", "Body2", "body1", "body2", "vag0",
                 "mask_Low", "mask_Top", "Feet", "Leg", "Hand'", "ARM",
                 "MaskNipple", "MaskBody"):
        obj.vertex_groups.new(name=name).add(list(range(12)), 1.0, "REPLACE")
    return obj


def armature_fixture(name, bone_names):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for index, bone_name in enumerate(sorted(bone_names)):
        bone = data.edit_bones.new(bone_name)
        bone.head = (0, 0, index)
        bone.tail = (0, 0, index + 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def test_table():
    expected = {}
    for side in ("L", "R"):
        lower = side.lower()
        for source, target in (
            ("arm_stretch", "UpArmTwist1"),
            ("c_arm_twist_offset", "UpArmTwist"),
            ("forearm_stretch", "_ForeTwist"),
            ("forearm_twist", "_ForeTwist1"),
            ("thigh_stretch", "ThighTwist1"),
            ("thigh_twist", "ThighTwist"),
            ("leg_stretch", "CalfTwist"),
            ("leg_twist", "CalfTwist1"),
        ):
            expected[f"{source}.{lower}"] = f"Bip001_{side}{target}"
    assert _build_mapping(AUTO_RIG_ZMD_MAPPING_ID, set(expected)) == expected
    assert rig_mapping_items(None, None) is rig_mapping_items(None, None)
    assert (AUTO_RIG_ZMD_MAPPING_ID, "autorig->zmd") in [
        item[:2] for item in rig_mapping_items(None, None)
    ]
    assert not _build_mapping(AUTO_RIG_ZMD_MAPPING_ID, {"Bn_l_breast_002"})
    print("ZMD_TABLE_OK")


def test_operator(breast_targets=("breast_base_L_a_01_jnt", "breast_base_R_a_01_jnt")):
    bpy.ops.object.select_all(action="DESELECT")
    first, second, untouched = [fixture(name) for name in ("First", "Second", "Untouched")]
    rig = bpy.data.objects.new("UnrelatedRig", bpy.data.armatures.new("UnrelatedRig"))
    bpy.context.collection.objects.link(rig)
    first.parent = rig
    first.modifiers.new("OriginalArmature", "ARMATURE").object = rig
    snapshots = [mesh_snapshot(o) for o in (first, second, untouched)]
    original_unselected = weights(untouched)
    left_breast, right_breast = breast_targets
    tiny_clavicle = 6.31e-8 / (0.590825 + 0.330055 + 6.31e-8)
    expected = [
        {left_breast: 0.6, "Bip001_Pelvis": 0.4},
        {right_breast: 0.5, "Bip001_Pelvis": 0.5},
        {"Bip001_LUpArmTwist1": 0.7, "Bip001_LUpArmTwist": 0.3},
        {"Bip001_R_ForeTwist": 0.4, "Bip001_R_ForeTwist1": 0.6},
        {"Bip001_L_Toe0": 0.5, "Bip001_L_Foot": 0.5},
        {"Bip001_Pelvis": 1.0},
        {"Bip001_L_Finger1": 0.2, "Bip001_L_Finger11": 0.3, "Bip001_L_Finger12": 0.5},
        {"Bip001_L_UpperArm_corrective_ty_plus": 0.2, "Bip001_Pelvis": 0.8},
        {left_breast: 1.0},
        {right_breast: 1.0},
        {left_breast: 1.0 - tiny_clavicle, "Bip001_L_Clavicle": tiny_clavicle},
        {right_breast: 1.0 - tiny_clavicle, "Bip001_R_Clavicle": tiny_clavicle},
    ]
    target = armature_fixture("TargetRig", {name for row in expected for name in row})
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = first
    settings = bpy.context.scene.mdt_group_match_settings
    settings.rig_mapping = AUTO_RIG_ZMD_MAPPING_ID
    settings.target_armature = target
    assert settings.source_object is None and settings.target_object is None
    assert bpy.ops.mdt.auto_rig_npc() == {"FINISHED"}
    for _repeat in range(2):
        for obj in (first, second):
            for actual, want in zip(weights(obj), expected):
                assert actual.keys() == want.keys(), (actual, want)
                assert all(abs(actual[n] - w) < 1.0e-6 for n, w in want.items())
        assert [mesh_snapshot(o) for o in (first, second, untouched)] == snapshots
        assert weights(untouched) == original_unselected
        assert bpy.ops.mdt.auto_rig_npc() == {"FINISHED"}
    print("ZMD_BATCH_MERGE_MASKS_REPEAT_GEOMETRY_OK", breast_targets)


def test_unmatched():
    mesh = bpy.data.meshes.new("Unsupported")
    mesh.from_pydata([(0, 0, 0)], [], [])
    obj = bpy.data.objects.new("Unsupported", mesh)
    obj.vertex_groups.new(name="MaskOnly").add([0], 1.0, "REPLACE")
    before = weights(obj)
    try:
        apply_auto_rig_npc(obj, AUTO_RIG_ZMD_MAPPING_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported groups must be rejected before mutation")
    assert weights(obj) == before
    print("ZMD_UNMATCHED_NO_MUTATION_OK")


def test_palm_and_finger_segments():
    # Palm, proximal, middle and distal samples must remain independent.
    source_rows, expected_rows = [], []
    for side in ("L", "R"):
        lower = side.lower()
        palm = {f"hand.{lower}": 0.2}
        palm.update({f"c_{finger}1_base.{lower}": 0.15
                     for finger in ("index", "middle", "ring", "pinky")})
        palm[f"forearm_twist.{lower}"] = 0.2
        source_rows.append(palm)
        expected_rows.append({f"Bip001_{side}_Hand": 0.8,
                              f"Bip001_{side}_ForeTwist1": 0.2})
        for finger, number in (("index", 1), ("middle", 2), ("ring", 3), ("pinky", 4)):
            for source, target in (
                (f"{finger}1.{lower}", f"Bip001_{side}_Finger{number}"),
                (f"c_{finger}2.{lower}", f"Bip001_{side}_Finger{number}1"),
                (f"c_{finger}3.{lower}", f"Bip001_{side}_Finger{number}2"),
            ):
                source_rows.append({source: 1.0})
                expected_rows.append({target: 1.0})
        for source, target in (("thumb1", "Finger0"), ("c_thumb2", "Finger01"),
                               ("c_thumb3", "Finger02")):
            source_rows.append({f"{source}.{lower}": 1.0})
            expected_rows.append({f"Bip001_{side}_{target}": 1.0})
    mesh = bpy.data.meshes.new("PalmAndFingers")
    mesh.from_pydata([(i, 0, 0) for i in range(len(source_rows))], [], [])
    obj = bpy.data.objects.new("PalmAndFingers", mesh)
    for index, row in enumerate(source_rows):
        for name, value in row.items():
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([index], value, "REPLACE")
    for _repeat in range(2):
        apply_auto_rig_npc(obj, AUTO_RIG_ZMD_MAPPING_ID)
        for index, (actual, expected) in enumerate(zip(weights(obj), expected_rows)):
            assert actual.keys() == expected.keys(), (index, actual, expected)
            assert all(abs(actual[n] - w) < 1.0e-6 for n, w in expected.items())
    print("ZMD_PALM_AND_FINGER_SEGMENTS_OK")


def test_vrc_names_preserve_physics_without_armatures():
    bpy.ops.object.select_all(action="DESELECT")
    rows = [
        {"Hips": 0.2, "Butt_L": 0.3, "Tummy Jiggle": 0.5},
        {"Chest": 0.2, "Boob_L": 0.3, "Chestdanglies_R.007": 0.5},
        {"Arm_L": 0.2, "Elbow_Twist_L": 0.3, "Silk_L.006": 0.5},
        {"Leg_R": 0.2, "Thigh Jiggle_R": 0.8},
        {"Index Finger 02_L": 0.4, "Index Finger 03_L": 0.6},
        {"Ring Finger _R": 0.2, "Ring Finger 02_R": 0.3, "Ring Finger 03_R": 0.5},
        {"Wrist_L": 0.3, "Thumb_L": 0.2, "Thumb 01_L": 0.2, "Thumb 03_L": 0.3},
        {"Hips": 0.2, "Hips Dips": 0.3, "Silkskirt_L.006": 0.5},
        {"Neckdanglies_M.001": 0.5, "Hipdanglies_L.023": 0.5},
    ]
    expected = [
        {"Bip001_Pelvis": 0.2, "Butt_L": 0.3, "Tummy Jiggle": 0.5},
        {"Bip001_Spine2": 0.2, "Boob_L": 0.3, "Chestdanglies_R.007": 0.5},
        {"Bip001_L_UpperArm": 0.2, "Elbow_Twist_L": 0.3, "Silk_L.006": 0.5},
        {"Bip001_R_Thigh": 0.2, "Thigh Jiggle_R": 0.8},
        {"Bip001_L_Finger11": 0.4, "Bip001_L_Finger12": 0.6},
        {"Bip001_R_Finger3": 0.2, "Bip001_R_Finger31": 0.3, "Bip001_R_Finger32": 0.5},
        {"Bip001_L_Hand": 0.3, "Bip001_L_Finger0": 0.2,
         "Bip001_L_Finger01": 0.2, "Bip001_L_Finger02": 0.3},
        {"Bip001_Pelvis": 0.2, "Hips Dips": 0.3, "Silkskirt_L.006": 0.5},
        {"Neckdanglies_M.001": 0.5, "Hipdanglies_L.023": 0.5},
    ]
    objects = []
    for name in ("VRCFirst", "VRCSecond", "VRCUntouched"):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(i % 3, i // 3, 0) for i in range(len(rows))], [],
                         [(0, 1, 4, 3), (1, 2, 5, 4)])
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        uv = mesh.uv_layers.new(name="OriginalUV")
        uv.data.foreach_set("uv", np.arange(len(mesh.loops) * 2, dtype=np.float32) / 10)
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="ExistingShape")
        key.data.foreach_set("co", dense(mesh.vertices, "co", 3) + 0.2)
        key.value = 0.6
        for index, row in enumerate(rows):
            for group_name, weight in row.items():
                group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
                group.add([index], weight, "REPLACE")
        obj.vertex_groups.new(name="MaskBody").add(list(range(len(rows))), 1.0, "REPLACE")
        obj.vertex_groups.new(name="EmptyPhysics").lock_weight = True
        obj.vertex_groups["Boob_L"].lock_weight = True
        objects.append(obj)
    first, second, untouched = objects
    snapshots = [mesh_snapshot(o) for o in objects]
    untouched_weights = weights(untouched)
    weights_by_index = [
        [[(a.group, a.weight) for a in v.groups] for v in o.data.vertices]
        for o in objects
    ]
    group_flags = [[(g.index, g.lock_weight) for g in o.vertex_groups] for o in objects]
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = first
    settings = bpy.context.scene.mdt_group_match_settings
    settings.target_armature = None
    settings.rig_mapping = VRC_ZMD_MAPPING_ID
    assert (VRC_ZMD_MAPPING_ID, "vrc->zmd") in [
        item[:2] for item in rig_mapping_items(None, None)
    ]
    for row in expected:
        row["MaskBody"] = 1.0
    for _repeat in range(2):
        assert bpy.ops.mdt.auto_rig_npc() == {"FINISHED"}
        for obj in (first, second):
            for actual, want in zip(weights(obj), expected):
                assert actual.keys() == want.keys(), (actual, want)
                assert all(abs(actual[n] - value) < 1.0e-6 for n, value in want.items())
                assert abs(sum(actual.values()) - 2.0) < 1.0e-6
        assert [mesh_snapshot(o) for o in objects] == snapshots
        assert weights(untouched) == untouched_weights
        assert [[[(a.group, a.weight) for a in v.groups] for v in o.data.vertices]
                for o in objects] == weights_by_index
        assert [[(g.index, g.lock_weight) for g in o.vertex_groups] for o in objects] == group_flags
        assert all("EmptyPhysics" in o.vertex_groups for o in objects)
    before = weights(first)
    try:
        apply_auto_rig_npc(first, VRC_ZMD_MAPPING_ID, target_bone_names={"Bip001_Pelvis"})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("An incomplete target rig must be rejected before mutation")
    assert weights(first) == before
    assert not _build_mapping(VRC_ZMD_MAPPING_ID, {"Boob_L", "Chestdanglies_R.001", "Silk_L"})
    collision_mesh = bpy.data.meshes.new("VRCConflict")
    collision_mesh.from_pydata([(0, 0, 0)], [], [])
    collision = bpy.data.objects.new("VRCConflict", collision_mesh)
    collision.vertex_groups.new(name="Hips").add([0], 0.4, "REPLACE")
    collision.vertex_groups.new(name="Bip001_Pelvis").add([0], 0.6, "REPLACE")
    before = weights(collision)
    try:
        apply_auto_rig_npc(collision, VRC_ZMD_MAPPING_ID)
    except ValueError as exc:
        assert "not merged" in str(exc)
    else:
        raise AssertionError("Name collisions must not silently merge or suffix groups")
    assert weights(collision) == before
    print("VRC_RENAME_PHYSICS_EXACT_WEIGHTS_LOCKS_BATCH_REPEAT_NO_RIG_OK")


if __name__ == "__main__":
    bpy.utils.register_class(MDT_PG_GroupMatchSettings)
    bpy.types.Scene.mdt_group_match_settings = bpy.props.PointerProperty(type=MDT_PG_GroupMatchSettings)
    bpy.utils.register_class(MDT_OT_auto_rig_npc)
    try:
        test_table()
        test_palm_and_finger_segments()
        test_operator()
        test_operator(("L_breast_01_jnt", "R_breast_01_jnt"))
        test_operator(("xiong_L_0_skin_jnt", "xiong_R_0_skin_jnt"))
        test_unmatched()
        test_vrc_names_preserve_physics_without_armatures()
        print("ZMD_GROUP_RENAME_SMOKE_OK")
    finally:
        bpy.utils.unregister_class(MDT_OT_auto_rig_npc)
        del bpy.types.Scene.mdt_group_match_settings
        bpy.utils.unregister_class(MDT_PG_GroupMatchSettings)
