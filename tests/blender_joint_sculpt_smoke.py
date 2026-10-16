"""Joint Sculpt regression checks in a separate Blender process."""

import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np
from mathutils import Matrix

import modtool
from modtool.core.common import ensure_object_mode
from modtool.core.joint_sculpt import apply_joint_sculpt


def coords(points):
    result = np.empty((len(points), 3), dtype=np.float32)
    points.foreach_get("co", result.ravel())
    return result


def visible(obj):
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    try:
        return coords(mesh.vertices)
    finally:
        ev.to_mesh_clear()


def reset():
    ensure_object_mode(bpy.context)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def fixture(shapes=False, absolute=False, transforms=True):
    reset()
    objects = []
    for index in range(2):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=9, y_subdivisions=9, size=1)
        obj = bpy.context.object
        obj.name = f"JointTest{index}"
        objects.append(obj)
        if transforms:
            obj.matrix_world = Matrix.Translation((index * .2, 0, index * .1)) @ Matrix.Rotation(.15 * index, 4, "Y") @ Matrix.Diagonal((1, (-1 if index else 1) * .8, 1.1, 1))
        else:
            obj.location.z = index * .025
        mesh = obj.data
        for name in ("UV0", "UV1"):
            layer = mesh.uv_layers.get(name) or mesh.uv_layers.new(name=name)
            values = np.empty((len(mesh.loops), 2), dtype=np.float32)
            indices = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", indices)
            values[:] = coords(mesh.vertices)[indices, :2] + .5
            if name == "UV1":
                values *= .4
            layer.data.foreach_set("uv", values.ravel())
        mesh.uv_layers["UV0" if index == 0 else "UV1"].active_render = True
        mat = bpy.data.materials.new(obj.name)
        mat.use_nodes = True
        mesh.materials.append(mat)
        group = obj.vertex_groups.new(name="TestWeight")
        group.add(list(range(len(mesh.vertices))), .75, "REPLACE")
        if shapes:
            obj.shape_key_add(name="Basis")
            first = obj.shape_key_add(name="Size")
            values = coords(first.data)
            values[:, 2] += .1
            first.data.foreach_set("co", values.ravel())
            first.value = .6
            second = obj.shape_key_add(name="Detail")
            values[:, 2] += .15
            second.data.foreach_set("co", values.ravel())
            second.relative_key = first
            second.vertex_group = group.name
            second.value = .3
            second.lock_shape = True
            obj.active_shape_key_index = 1
            if absolute:
                mesh.shape_keys.use_relative = False
                mesh.shape_keys.eval_time = 5
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.context.view_layer.update()
    return objects


def preservation(obj):
    uv = []
    for layer in obj.data.uv_layers:
        values = np.empty(len(layer.data) * 2, dtype=np.float32)
        layer.data.foreach_get("uv", values)
        uv.append((layer.name, values.tobytes()))
    return (uv, [s.material.as_pointer() for s in obj.material_slots], [[(g.group, g.weight) for g in v.groups] for v in obj.data.vertices])


def test_apply(shapes=False, absolute=False):
    objects = fixture(shapes, absolute)
    state = [preservation(obj) for obj in objects]
    original = [coords(obj.data.vertices) for obj in objects]
    evaluated = [visible(obj) for obj in objects]
    key_coordinates = [[coords(k.data) for k in obj.data.shape_keys.key_blocks] if shapes else [] for obj in objects]
    shape_values = [[(k.value, k.mute, k.lock_shape, k.relative_key.name, k.vertex_group) for k in obj.data.shape_keys.key_blocks] if shapes else [] for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    proxy = settings.proxy
    assert bpy.context.mode == "SCULPT" and all(obj.hide_get() for obj in objects)
    assert len(proxy.data.vertices) == sum(len(o.data.vertices) for o in objects)
    proxy_inverse = np.linalg.inv(np.asarray(proxy.matrix_world))
    start = coords(proxy.data.vertices)
    for row, expected in zip(settings.sources, evaluated):
        matrix = proxy_inverse @ np.asarray(row.object.matrix_world)
        np.testing.assert_allclose(start[row.start:row.start + row.count], expected @ matrix[:3, :3].T + matrix[:3, 3], atol=2e-6)
    # Exercise world/local mapping and every shape-key coordinate on both objects.
    ensure_object_mode(bpy.context)
    displacement = np.array((.025, -.015, .035), dtype=np.float32)
    proxy.data.vertices.foreach_set("co", (start + displacement).ravel())
    matrix = np.asarray(proxy.matrix_world)[:3, :3]
    assert bpy.ops.mdt.apply_joint_sculpt() == {"FINISHED"}
    assert bpy.context.mode == "OBJECT" and bpy.context.object == objects[0]
    assert all(not obj.hide_get() for obj in objects)
    bpy.context.view_layer.update()
    for i, obj in enumerate(objects):
        delta = displacement @ (np.linalg.inv(np.asarray(obj.matrix_world))[:3, :3] @ matrix).T
        np.testing.assert_allclose(coords(obj.data.vertices), original[i] + delta, atol=2e-6)
        np.testing.assert_allclose(visible(obj), evaluated[i] + delta, atol=3e-6)
        assert preservation(obj) == state[i]
        if shapes:
            assert shape_values[i] == [(k.value, k.mute, k.lock_shape, k.relative_key.name, k.vertex_group) for k in obj.data.shape_keys.key_blocks]
            for key, previous in zip(obj.data.shape_keys.key_blocks, key_coordinates[i]):
                np.testing.assert_allclose(coords(key.data), previous + delta, atol=2e-6)
    assert not settings.sources and settings.proxy is None
    assert settings.previous_active is None and settings.proxy_mesh is None
    print("APPLY_OK", shapes, absolute)


def test_failures_and_cancel():
    objects = fixture(shapes=True)
    original = [coords(obj.data.vertices) for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    ensure_object_mode(bpy.context)
    # Same-size remeshing/reordering must also be rejected, not just changed counts.
    ids = settings.proxy.data.attributes[".mdt_joint_sculpt_id"]
    ids.data[0].value = 7
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("correspondence mutation accepted")
    except ValueError as exc:
        assert "correspondence" in str(exc)
    for obj, previous in zip(objects, original):
        np.testing.assert_array_equal(coords(obj.data.vertices), previous)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    assert all(not obj.hide_get() for obj in objects)
    assert settings.previous_active is None and settings.proxy_mesh is None
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    settings.proxy.data.vertices.add(1)
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("new topology accepted")
    except ValueError as exc:
        assert "topology" in str(exc)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    bpy.data.objects.remove(settings.proxy, do_unlink=True)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    assert all(not obj.hide_get() for obj in objects)
    for obj, previous in zip(objects, original):
        np.testing.assert_array_equal(coords(obj.data.vertices), previous)
    print("CANCEL_AND_FAILURES_OK")


def test_save_reload():
    fixture(shapes=True)
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "session.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
        settings = bpy.context.scene.mdt_joint_sculpt_settings
        assert settings.proxy and len(settings.sources) == 2
        assert bpy.ops.mdt.apply_joint_sculpt() == {"FINISHED"}
    print("SAVE_RELOAD_OK")


def test_armature_and_atomic_validation():
    objects = fixture(shapes=True)
    rig_data = bpy.data.armatures.new("JointTestRig")
    rig = bpy.data.objects.new("JointTestRig", rig_data)
    bpy.context.collection.objects.link(rig)
    for obj in objects:
        obj.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bone = rig_data.edit_bones.new("Bone")
    bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)
    for obj in objects:
        modifier = obj.modifiers.new("TestArmature", "ARMATURE")
        modifier.object = rig
        group = obj.vertex_groups.new(name="Bone")
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.context.view_layer.update()
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    previous = coords(objects[0].data.vertices)
    keys = [coords(k.data) for k in objects[0].data.shape_keys.key_blocks]
    proxy_positions = coords(settings.proxy.data.vertices) + (.01, .02, .03)
    settings.proxy.data.vertices.foreach_set("co", proxy_positions.ravel())
    objects[1].data.shape_keys.key_blocks[1].value = .2
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("Changed source shape-key state accepted")
    except ValueError as exc:
        assert "state changed" in str(exc)
    np.testing.assert_array_equal(coords(objects[0].data.vertices), previous)
    for key, expected in zip(objects[0].data.shape_keys.key_blocks, keys):
        np.testing.assert_array_equal(coords(key.data), expected)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    rig.pose.bones["Bone"].rotation_mode = "XYZ"
    rig.pose.bones["Bone"].rotation_euler.x = .3
    bpy.context.view_layer.update()
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    assert all(not obj.hide_get() for obj in objects)
    assert settings.proxy is None and not settings.sources
    print("ARMATURE_AND_ATOMIC_VALIDATION_OK")


def posed_fixture(shapes=True, absolute=False, preserve_volume=False):
    objects = fixture(shapes=shapes, absolute=absolute)
    for obj in objects:
        obj.select_set(False)
    rig_data = bpy.data.armatures.new("PosedRig")
    rig = bpy.data.objects.new("PosedRig", rig_data)
    bpy.context.collection.objects.link(rig)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, x in (("A", -.3), ("B", .3)):
        bone = rig_data.edit_bones.new(name)
        bone.head, bone.tail = (x, 0, -.5), (x, 0, .5)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)
    for index, name in enumerate(("A", "B")):
        bone = rig.pose.bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (.45, -.25 * index, .3 * (index + 1))
        bone.scale = (1.2, .8, 1.4) if index == 0 else (.9, 1.15, .75)
        bone.location = (.05, .03 * index, -.04)
    for obj in objects:
        a, b = (obj.vertex_groups.new(name=name) for name in ("A", "B"))
        for vertex in obj.data.vertices:
            weight = .1 + .8 * vertex.index / (len(obj.data.vertices) - 1)
            a.add([vertex.index], weight, "REPLACE")
            b.add([vertex.index], 1 - weight, "REPLACE")
        modifier = obj.modifiers.new("PosedArmature", "ARMATURE")
        modifier.object = rig
        modifier.use_deform_preserve_volume = preserve_volume
        modifier.vertex_group = "TestWeight"
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.context.view_layer.update()
    return objects, rig


def test_posed_apply(shapes=True, absolute=False, preserve_volume=False, active_zero=False, show_only=False):
    objects, rig = posed_fixture(shapes, absolute, preserve_volume)
    if active_zero:
        for obj in objects:
            obj.data.shape_keys.key_blocks[1].value = 0
    if show_only:
        for obj in objects:
            obj.show_only_shape_key = True
    bpy.context.view_layer.update()
    before = [preservation(obj) for obj in objects]
    originals = [coords(obj.data.vertices) for obj in objects]
    key_coords = [[coords(k.data) for k in obj.data.shape_keys.key_blocks] if shapes else [] for obj in objects]
    visible_before = [visible(obj) for obj in objects]
    indices = [obj.active_shape_key_index for obj in objects]
    modifiers = [[m.as_pointer() for m in obj.modifiers] for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    proxy = settings.proxy
    start = coords(proxy.data.vertices)
    frame = np.asarray(proxy.matrix_world)
    for row, expected in zip(settings.sources, visible_before):
        transform = np.linalg.inv(frame) @ np.asarray(row.object.matrix_world)
        np.testing.assert_allclose(start[row.start:row.start + row.count], expected @ transform[:3, :3].T + transform[:3, 3], atol=2e-6)
    ensure_object_mode(bpy.context)
    displacement = np.zeros_like(start)
    displacement[::3] = (.023, -.017, .031)
    proxy.data.vertices.foreach_set("co", (start + displacement).ravel())
    rows = [(row.start, row.count) for row in settings.sources]
    assert bpy.ops.mdt.apply_joint_sculpt() == {"FINISHED"}
    bpy.context.view_layer.update()
    for index, (obj, (offset, count)) in enumerate(zip(objects, rows)):
        transform = np.linalg.inv(np.asarray(obj.matrix_world))[:3, :3] @ frame[:3, :3]
        expected = visible_before[index] + displacement[offset:offset + count] @ transform.T
        np.testing.assert_allclose(visible(obj), expected, atol=4e-6)
        assert preservation(obj) == before[index]
        assert obj.active_shape_key_index == indices[index]
        assert [m.as_pointer() for m in obj.modifiers] == modifiers[index]
        delta = coords(obj.data.vertices) - originals[index]
        for key, old in zip(obj.data.shape_keys.key_blocks if shapes else [], key_coords[index]):
            np.testing.assert_allclose(coords(key.data), old + delta, atol=2e-6)
    print("POSED_APPLY_OK", shapes, absolute, preserve_volume, active_zero, show_only)


def test_pose_change_rejected():
    objects, rig = posed_fixture()
    original = [coords(obj.data.vertices) for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    rig.pose.bones["A"].rotation_euler.x += .2
    bpy.context.view_layer.update()
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("Changed pose accepted")
    except ValueError as exc:
        assert "deformation changed" in str(exc)
    for obj, old in zip(objects, original):
        np.testing.assert_array_equal(coords(obj.data.vertices), old)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    print("POSE_CHANGE_REJECTED_OK")


def test_modifier_change_rejected():
    objects, _rig = posed_fixture()
    originals = [coords(obj.data.vertices) for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    objects[1].modifiers[0].show_viewport = False
    bpy.context.view_layer.update()
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("Disabled armature accepted")
    except ValueError as exc:
        assert "state changed" in str(exc)
    for obj, previous in zip(objects, originals):
        np.testing.assert_array_equal(coords(obj.data.vertices), previous)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    print("MODIFIER_CHANGE_REJECTED_OK")


def test_singular_deformation_rejected():
    objects, rig = posed_fixture()
    for bone in rig.pose.bones:
        bone.scale = (0, 0, 0)
    objects[0].modifiers[0].show_viewport = False
    objects[1].modifiers[0].vertex_group = ""
    bpy.context.view_layer.update()
    original = [[coords(obj.data.vertices)] + [coords(k.data) for k in obj.data.shape_keys.key_blocks] for obj in objects]
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    ensure_object_mode(bpy.context)
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    settings.proxy.data.vertices.foreach_set("co", (coords(settings.proxy.data.vertices) + (.01, .02, .03)).ravel())
    object_count, mesh_count = len(bpy.data.objects), len(bpy.data.meshes)
    try:
        apply_joint_sculpt(bpy.context)
        raise AssertionError("Singular bone scale accepted")
    except ValueError as exc:
        assert "cannot invert" in str(exc)
    assert (len(bpy.data.objects), len(bpy.data.meshes)) == (object_count, mesh_count)
    assert settings.proxy and len(settings.sources) == 2
    for obj, previous in zip(objects, original):
        for points, expected in zip([obj.data.vertices] + [k.data for k in obj.data.shape_keys.key_blocks], previous):
            np.testing.assert_array_equal(coords(points), expected)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    print("SINGULAR_DEFORMATION_ATOMIC_OK")


def test_native_brush():
    objects = fixture(transforms=False)
    bpy.ops.ed.undo_push(message="Initialize native sculpt test")
    assert bpy.ops.mdt.start_joint_sculpt() == {"FINISHED"}
    settings = bpy.context.scene.mdt_joint_sculpt_settings
    bpy.ops.brush.asset_activate(asset_library_type="ESSENTIALS", relative_asset_identifier="brushes/essentials_brushes-mesh_sculpt.blend/Brush/Draw")
    brush = bpy.context.scene.tool_settings.sculpt.brush
    brush.size = 200
    brush.strength = .5
    brush.use_automasking_topology = False
    brush.use_automasking_face_sets = False
    before = coords(settings.proxy.data.vertices)
    area = next(a for a in bpy.context.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    area.spaces.active.region_3d.view_rotation = Matrix.Identity(3).to_quaternion()
    area.spaces.active.region_3d.view_distance = 2
    stroke = [{"name": "", "location": (0, 0, .025), "mouse": (region.width / 2, region.height / 2), "mouse_event": (region.width / 2, region.height / 2), "pressure": 1, "size": 200, "x_tilt": 0, "y_tilt": 0, "time": i * .1, "is_start": i == 0} for i in range(4)]
    with bpy.context.temp_override(area=area, region=region):
        assert bpy.ops.sculpt.brush_stroke(stroke=stroke, override_location=False) == {"FINISHED"}
    after = coords(settings.proxy.data.vertices)
    changed = [int(np.count_nonzero(np.any(after[r.start:r.start + r.count] != before[r.start:r.start + r.count], axis=1))) for r in settings.sources]
    assert all(count > 0 for count in changed), changed
    if not bpy.app.background:
        with bpy.context.temp_override(area=area, region=region):
            assert bpy.ops.ed.undo() == {"FINISHED"}
        settings = bpy.context.scene.mdt_joint_sculpt_settings
        assert settings.proxy and len(settings.sources) == 2
        np.testing.assert_array_equal(coords(settings.proxy.data.vertices), before)
        print("NATIVE_SCULPT_UNDO_OK")
    ensure_object_mode(bpy.context)
    assert bpy.ops.mdt.cancel_joint_sculpt() == {"FINISHED"}
    print("NATIVE_BRUSH_BOTH_OBJECTS_OK", changed)


modtool.register()
test_apply()
test_apply(shapes=True)
test_apply(shapes=True, absolute=True)
test_failures_and_cancel()
test_armature_and_atomic_validation()
test_posed_apply(shapes=False)
test_posed_apply()
test_posed_apply(preserve_volume=True)
test_posed_apply(absolute=True)
test_posed_apply(active_zero=True)
test_posed_apply(show_only=True)
test_pose_change_rejected()
test_modifier_change_rejected()
test_singular_deformation_rejected()
test_save_reload()
test_native_brush()
print("JOINT_SCULPT_SMOKE_OK")
