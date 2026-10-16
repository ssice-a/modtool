"""Cycles integration checks for color mixing and per-material bake UVs."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np

import modtool
from modtool.core.material_bake import bake_output_colors


def read_uv(layer):
    result = np.empty(len(layer.data) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", result)
    return result


def build_object():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new("ColorBakeTest")
    vertices = [(x + i * 2, y, 0) for i in range(3) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    mesh.from_pydata(vertices, [], [(i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3) for i in range(3)])
    obj = bpy.data.objects.new("ColorBakeTest", mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    uv0 = mesh.uv_layers.new(name="UV0")
    uv1 = mesh.uv_layers.new(name="UV1")
    boxes = [(0.05, 0.3), (0.35, 0.6), (0.65, 0.9)]
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for index, polygon in enumerate(mesh.polygons):
        for loop, (u, v) in zip(polygon.loop_indices, corners):
            uv0.data[loop].uv = (u, v) if index == 0 else (boxes[index][0] + u * .25, .05 + v * .4)
            uv1.data[loop].uv = (boxes[index][0] + u * .25, .05 + v * .4)
    mesh.uv_layers.active_index = 0
    uv0.active_render = True

    image = bpy.data.images.new("SamplingUVTest", width=8, height=8)
    image.colorspace_settings.name = "Non-Color"
    pixels = np.ones((8, 8, 4), dtype=np.float32)
    pixels[:, :4, :3] = (.8, .1, .2)
    pixels[:, 4:, :3] = (.1, .8, .2)
    image.pixels.foreach_set(pixels.ravel())
    image.update()

    # Nested group with an unused EEVEE output: pruning must follow the chosen color.
    inner = bpy.data.node_groups.new("ColorBakeInner", "ShaderNodeTree")
    inner.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    inner.interface.new_socket(name="Unused lighting", in_out="OUTPUT", socket_type="NodeSocketColor")
    tex = inner.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Closest"
    out = inner.nodes.new("NodeGroupOutput")
    inner.links.new(tex.outputs["Color"], out.inputs[0])
    lit = inner.nodes.new("ShaderNodeShaderToRGB")
    inner.links.new(lit.outputs["Color"], out.inputs[1])
    outer = bpy.data.node_groups.new("ColorBakeOuter", "ShaderNodeTree")
    outer.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    child = outer.nodes.new("ShaderNodeGroup")
    child.node_tree = inner
    out = outer.nodes.new("NodeGroupOutput")
    outer.links.new(child.outputs[0], out.inputs[0])

    for index in range(3):
        mat = bpy.data.materials.new(f"ColorBakeMat{index}")
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        if index == 0:
            color = tree.nodes.new("ShaderNodeGroup")
            color.node_tree = outer
            socket = color.outputs[0]
        else:
            color = tree.nodes.new("ShaderNodeRGB")
            color.outputs[0].default_value = ((.2, .3, .8, 1) if index == 1 else (.8, .5, .1, 1))
            socket = color.outputs[0]
        tree.links.new(socket, output.inputs["Surface"])
        mesh.materials.append(mat)
        mesh.polygons[index].material_index = index
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="TestKey")
    key.data[0].co.z = .2
    key.value = .4
    return obj


def counts():
    return tuple(len(items) for items in (bpy.data.scenes, bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images))


def main():
    modtool.register()
    obj = build_object()
    original_uvs = [read_uv(u) for u in obj.data.uv_layers]
    original_keys = np.array([tuple(p.co) for p in obj.data.shape_keys.key_blocks[1].data])
    engine = bpy.context.scene.render.engine
    with tempfile.TemporaryDirectory() as directory:
        settings = SimpleNamespace(mesh_object=obj, source_uv="UV0", resolution="64", margin=0, filepath=str(Path(directory) / "color.png"), materials=[SimpleNamespace(slot_index=i, material=slot.material, target_uv="UV1" if i == 0 else "UV0") for i, slot in enumerate(obj.material_slots)])
        baseline = counts()
        result = bake_output_colors(bpy.context, obj, settings)
        pixels = np.empty(64 * 64 * 4, dtype=np.float32)
        result["image"].pixels.foreach_get(pixels)
        pixels = pixels.reshape(64, 64, 4)
        # Left and right halves of material 0's tiny destination island still sample
        # the full original UV0. Sampling the bake UV instead would make both red.
        for xy, expected in [((7, 15), (.8, .1, .2)), ((15, 15), (.1, .8, .2)), ((29, 15), (.2, .3, .8)), ((49, 15), (.8, .5, .1))]:
            x, y = xy
            np.testing.assert_allclose(pixels[y, x, :3], expected, atol=.015)
        assert Path(settings.filepath).is_file()
        assert counts() == baseline[:-1] + (baseline[-1] + 1,), (baseline, counts())
        assert bpy.context.object == obj and bpy.context.scene.render.engine == engine
        assert obj.data.uv_layers.active.name == "UV0" and obj.data.uv_layers["UV0"].active_render
        for layer, expected in zip(obj.data.uv_layers, original_uvs):
            np.testing.assert_array_equal(read_uv(layer), expected)
        np.testing.assert_array_equal(np.array([tuple(p.co) for p in obj.data.shape_keys.key_blocks[1].data]), original_keys)
        assert obj.data.shape_keys.key_blocks[1].value == np.float32(.4)
        assert [p.material_index for p in obj.data.polygons] == [0, 1, 2]
        assert len(bpy.data.node_groups["ColorBakeInner"].nodes) == 3
        for mat in obj.data.materials:
            output = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
            assert output.inputs["Surface"].links[0].from_socket.type == "RGBA"

        # Re-read the saved PNG and bake it back using an explicit composite UV.
        # This catches incorrect PNG color-space conversion as well as UV binding.
        loaded = bpy.data.images.load(settings.filepath, check_existing=False)
        original_materials = list(obj.data.materials)
        roundtrip_mat = bpy.data.materials.new("SavedPNGCheck")
        roundtrip_mat.use_nodes = True
        tree = roundtrip_mat.node_tree
        tree.nodes.clear()
        uv = tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = result["uv_name"]
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = loaded
        tex.interpolation = "Closest"
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        tree.links.new(tex.outputs["Color"], output.inputs["Surface"])
        for slot in obj.material_slots:
            slot.material = roundtrip_mat
        for row in settings.materials:
            row.material = roundtrip_mat
        settings.filepath = str(Path(directory) / "roundtrip.png")
        roundtrip = bake_output_colors(bpy.context, obj, settings)
        reloaded_pixels = np.empty_like(pixels)
        roundtrip["image"].pixels.foreach_get(reloaded_pixels.ravel())
        for x, y in [(7, 15), (15, 15), (29, 15), (49, 15)]:
            np.testing.assert_allclose(reloaded_pixels[y, x, :3], pixels[y, x, :3], atol=.009)
        for slot, material in zip(obj.material_slots, original_materials):
            slot.material = material
        for row, material in zip(settings.materials, original_materials):
            row.material = material
        bpy.data.materials.remove(roundtrip_mat)
        bpy.data.images.remove(loaded)

        # Failure paths must not leave any temporary datablocks or new UVs behind.
        baseline = counts()
        for case in ("empty", "overlap", "shader"):
            settings.filepath = str(Path(directory) / f"{case}.png")
            saved_uv = read_uv(obj.data.uv_layers["UV1"])
            if case == "empty":
                obj.data.uv_layers["UV1"].data.foreach_set("uv", np.zeros_like(saved_uv))
            elif case == "overlap":
                settings.materials[0].target_uv = "UV0"
                obj.data.uv_layers["UV0"].data.foreach_set("uv", np.tile((0, 0, 1, 0, 1, 1, 0, 1), 3))
            else:
                mat = obj.data.materials[0]
                output = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
                shader = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                mat.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
            try:
                bake_output_colors(bpy.context, obj, settings)
                raise AssertionError(f"Expected {case} validation error")
            except ValueError as exc:
                print(case, str(exc))
            finally:
                obj.data.uv_layers["UV1"].data.foreach_set("uv", saved_uv)
                obj.data.uv_layers["UV0"].data.foreach_set("uv", original_uvs[0])
                settings.materials[0].target_uv = "UV1"
            assert not Path(settings.filepath).exists()
            assert counts() == baseline
    print("OUTPUT_COLOR_BAKE_SMOKE_OK")


main()
