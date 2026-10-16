"""Background round-trip test for Substance Painter Base Color preparation."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import modtool  # noqa: E402
from modtool.core.materials import (  # noqa: E402
    merge_base_color_for_sp,
    restore_base_color_from_sp,
)


def _reset():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)


def _image(name, width, height, color):
    image = bpy.data.images.new(name, width=width, height=height, alpha=True)
    pixels = np.empty((height, width, 4), dtype=np.float32)
    pixels[:, :, :] = color
    image.pixels.foreach_set(pixels.reshape(-1))
    image.colorspace_settings.name = "sRGB"
    image.update()
    return image


def _material(name, image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = image
    material.node_tree.links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
    return material


def _make_object():
    mesh = bpy.data.meshes.new("SPRoundTripMesh")
    mesh.from_pydata(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    mesh.update()
    obj = bpy.data.objects.new("SPRoundTrip", mesh)
    bpy.context.collection.objects.link(obj)
    first = _material("Wide", _image("WideColor", 4, 2, (1.0, 0.0, 0.0, 1.0)))
    second = _material("Tall", _image("TallColor", 2, 4, (0.0, 1.0, 0.0, 1.0)))
    unused = _material("Unused", _image("UnusedColor", 1, 1, (0.0, 0.0, 1.0, 1.0)))
    first.use_fake_user = False
    second.use_fake_user = True
    mesh.materials.append(first)
    mesh.materials.append(second)
    mesh.materials.append(unused)
    mesh.materials.append(first)
    mesh.polygons[0].material_index = 0
    mesh.polygons[1].material_index = 1
    uv_layer = mesh.uv_layers.new(name="UVMap")
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for polygon in mesh.polygons:
        for corner, loop_index in zip(corners, polygon.loop_indices):
            uv_layer.data[loop_index].uv = corner
    return obj, first, second, unused


def _uv_array(mesh):
    values = np.empty(len(mesh.uv_layers.active.data) * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", values)
    return values.reshape((-1, 2))


def _run():
    _reset()
    obj, first, second, unused = _make_object()
    original_uv = _uv_array(obj.data).copy()
    directory = tempfile.mkdtemp(prefix="mdt_sp_base_color_")
    result = merge_base_color_for_sp(obj, directory, 2)
    assert result["material_count"] == 3
    assert (result["atlas_width"], result["atlas_height"]) == (8, 8)
    assert Path(result["filepath"]).is_file()
    assert len(obj.data.materials) == 1
    assert [polygon.material_index for polygon in obj.data.polygons] == [0, 0]
    assert obj.get(".mdt_sp_base_color")
    assert obj.data.attributes.get(".mdt_sp_original_material_index") is not None
    merged_uv = _uv_array(obj.data)
    expected = np.asarray(
        (
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
            (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5),
        ),
        dtype=np.float32,
    )
    assert np.allclose(merged_uv, expected, atol=1.0e-6)

    atlas = bpy.data.images[result["image_name"]] if "image_name" in result else next(
        image for image in bpy.data.images if image.filepath_raw == result["filepath"]
    )
    pixels = np.empty(len(atlas.pixels), dtype=np.float32)
    atlas.pixels.foreach_get(pixels)
    pixels = pixels.reshape((8, 8, 4))
    pixels[0:4, 0:4] = (0.0, 0.0, 1.0, 1.0)
    pixels[0:4, 4:8] = (1.0, 1.0, 0.0, 1.0)
    atlas.pixels.foreach_set(pixels.reshape(-1))
    atlas.save()

    restored = restore_base_color_from_sp(obj)
    assert restored["material_count"] == 3
    assert list(obj.data.materials) == [first, second, unused, first]
    assert [polygon.material_index for polygon in obj.data.polygons] == [0, 1]
    assert np.allclose(_uv_array(obj.data), original_uv, atol=1.0e-6)
    assert obj.get(".mdt_sp_base_color") is None
    assert obj.data.attributes.get(".mdt_sp_original_material_index") is None
    assert first.use_fake_user is False
    assert second.use_fake_user is True
    assert len(restored["output_paths"]) == 3
    assert all(Path(path).is_file() for path in restored["output_paths"])
    first_node = next(node for node in first.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
    second_node = next(node for node in second.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
    unused_node = next(node for node in unused.node_tree.nodes if node.type == "TEX_IMAGE" and node.image)
    assert tuple(first_node.image.size) == (4, 4)
    assert tuple(second_node.image.size) == (4, 4)
    assert tuple(unused_node.image.size) == (4, 4)
    first_pixels = np.empty(len(first_node.image.pixels), dtype=np.float32)
    second_pixels = np.empty(len(second_node.image.pixels), dtype=np.float32)
    first_node.image.pixels.foreach_get(first_pixels)
    second_node.image.pixels.foreach_get(second_pixels)
    assert math.isclose(float(first_pixels[2]), 1.0, abs_tol=0.02)
    assert math.isclose(float(second_pixels[0]), 1.0, abs_tol=0.02)
    assert math.isclose(float(second_pixels[1]), 1.0, abs_tol=0.02)
    print("SP_BASE_COLOR_ROUND_TRIP_OK")


def _registration():
    modtool.register()
    try:
        assert hasattr(bpy.ops.mdt, "merge_base_color_for_sp")
        assert hasattr(bpy.ops.mdt, "restore_base_color_from_sp")
    finally:
        modtool.unregister()
    print("SP_BASE_COLOR_REGISTRATION_OK")


_run()
_registration()
