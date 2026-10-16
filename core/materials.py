"""Material helpers and reversible Base Color preparation for Substance Painter."""

from __future__ import annotations

import json
import os
import re
import uuid
from array import array
from dataclasses import dataclass

import bpy

from .common import mesh_topology_signature as _mesh_topology_signature


try:
    import numpy as np
except ImportError:  # pragma: no cover - Blender ships NumPy in supported builds.
    np = None


_SP_METADATA_KEY = ".mdt_sp_base_color"
_SP_ORIGINAL_INDEX_ATTRIBUTE = ".mdt_sp_original_material_index"
_SP_MATERIAL_ID_KEY = ".mdt_sp_material_id"
_SP_PREVIOUS_FAKE_USER_KEY = ".mdt_sp_previous_fake_user"
_MAX_ATLAS_PIXELS = 64 * 1024 * 1024


@dataclass(frozen=True)
class _BaseColorSource:
    material: object
    material_index: int
    image: object | None
    width: int
    height: int
    color: tuple[float, float, float, float]


def _read_polygon_material_indices(mesh):
    values = array("i", [0]) * len(mesh.polygons)
    if values:
        mesh.polygons.foreach_get("material_index", values)
    return values


def assign_single_material_to_object(obj, material):
    """Replace all material slots with one material and assign it to every polygon."""
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(material)
    if mesh.polygons:
        mesh.polygons.foreach_set("material_index", array("i", [0]) * len(mesh.polygons))
        mesh.update()


def resolve_used_material(obj):
    """Return the first material actually used by this mesh object's polygons."""
    for material_index in sorted(set(_read_polygon_material_indices(obj.data))):
        if 0 <= material_index < len(obj.material_slots):
            material = obj.material_slots[material_index].material
            if material is not None:
                return material
    return next((slot.material for slot in obj.material_slots if slot.material is not None), None)


def rename_object_from_material(obj):
    material = resolve_used_material(obj)
    if material is None:
        return False
    assign_single_material_to_object(obj, material)
    obj.name = material.name
    obj.data.name = f"{obj.name}_Mesh"
    return True


def _require_numpy(feature="Base Color merge"):
    if np is None:
        raise RuntimeError(f"{feature} requires NumPy, which is normally included with Blender.")


def _find_principled_base_color_input(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    output = next(
        (node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    )
    if output is not None:
        surface = output.inputs.get("Surface")
        if surface is not None:
            for link in surface.links:
                node = link.from_node
                if node.type == "BSDF_PRINCIPLED":
                    return node.inputs.get("Base Color")
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node.inputs.get("Base Color")
    return None


def _find_image_texture_node(socket, visited=None):
    if socket is None:
        return None
    if visited is None:
        visited = set()
    for link in socket.links:
        node = link.from_node
        pointer = node.as_pointer()
        if pointer in visited:
            continue
        visited.add(pointer)
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node
        for input_socket in node.inputs:
            image_node = _find_image_texture_node(input_socket, visited)
            if image_node is not None:
                return image_node
    return None


def _base_color_image_node(material):
    base_color = _find_principled_base_color_input(material)
    image_node = _find_image_texture_node(base_color) if base_color is not None else None
    if image_node is not None or material is None or material.node_tree is None:
        return image_node
    output = next(
        (node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    )
    return _find_image_texture_node(output.inputs.get("Surface")) if output is not None else None


def _base_color_value(material):
    input_socket = _find_principled_base_color_input(material)
    if input_socket is not None and hasattr(input_socket, "default_value"):
        return tuple(float(channel) for channel in input_socket.default_value[:4])
    return tuple(float(channel) for channel in material.diffuse_color[:4]) if material else (1.0, 1.0, 1.0, 1.0)


def _used_material_indices(mesh):
    return sorted(set(_read_polygon_material_indices(mesh)))


def _collect_sources(obj):
    mesh = obj.data
    used_indices = _used_material_indices(mesh)
    if not used_indices:
        raise ValueError(f"{obj.name} has no polygons using materials.")
    if any(index < 0 or index >= len(mesh.materials) for index in used_indices):
        raise ValueError(f"{obj.name} has faces referring to missing material slots.")
    if not mesh.materials:
        raise ValueError(f"{obj.name} has no materials.")
    sources = []
    image_sizes = []
    for material_index, material in enumerate(mesh.materials):
        if material is None:
            raise ValueError(f"Material slot {material_index} on {obj.name} is empty.")
        image_node = _base_color_image_node(material)
        image = image_node.image if image_node is not None else None
        if image is None and _find_principled_base_color_input(material) is None:
            raise ValueError(
                f"Material {material.name} needs a Principled Base Color input or a reachable image texture."
            )
        if image is not None:
            width, height = int(image.size[0]), int(image.size[1])
            if width <= 0 or height <= 0:
                raise ValueError(f"Base Color image {image.name} has no pixel data.")
            image_sizes.append((width, height))
        sources.append(
            _BaseColorSource(
                material=material,
                material_index=material_index,
                image=image,
                width=0,
                height=0,
                color=_base_color_value(material),
            )
        )
    if not image_sizes:
        raise ValueError("At least one used material needs a Base Color image.")
    fallback_width = max(width for width, _height in image_sizes)
    fallback_height = max(height for _width, height in image_sizes)
    return [
        _BaseColorSource(
            material=source.material,
            material_index=source.material_index,
            image=source.image,
            width=int(source.image.size[0]) if source.image is not None else fallback_width,
            height=int(source.image.size[1]) if source.image is not None else fallback_height,
            color=source.color,
        )
        for source in sources
    ]


def _read_image_pixels(image):
    width, height = int(image.size[0]), int(image.size[1])
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape((height, width, 4))


def _read_resampled_image_pixels(image, width, height):
    if tuple(image.size) == (width, height):
        return _read_image_pixels(image)
    scaled = image.copy()
    try:
        # Blender performs the resize in native code and retains scene-linear pixels.
        scaled.scale(width, height)
        return _read_image_pixels(scaled)
    finally:
        if scaled.users == 0:
            bpy.data.images.remove(scaled)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "Material"


def _resolve_directory(directory):
    if not directory:
        raise ValueError("Choose an output directory for the Base Color atlas.")
    resolved = bpy.path.abspath(directory)
    os.makedirs(resolved, exist_ok=True)
    if not os.path.isdir(resolved):
        raise ValueError(f"Base Color output path is not a directory: {resolved}")
    return resolved


def _create_saved_image(name, width, height, pixels, filepath):
    image = bpy.data.images.new(name=name, width=width, height=height, alpha=True)
    try:
        image.colorspace_settings.name = "sRGB"
        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()
        image.filepath_raw = filepath
        image.file_format = "PNG"
        image.save()
        return image
    except Exception:
        if image.users == 0:
            bpy.data.images.remove(image)
        raise


def _ensure_base_color_image_node(material, image, node_name):
    material.use_nodes = True
    base_color = _find_principled_base_color_input(material)
    if base_color is None:
        principled = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        base_color = principled.inputs["Base Color"]
    image_node = _base_color_image_node(material)
    if image_node is None:
        image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        image_node.name = node_name
        material.node_tree.links.new(image_node.outputs["Color"], base_color)
    image_node.image = image
    return image_node


def _ensure_original_index_attribute(mesh, values):
    if mesh.attributes.get(_SP_ORIGINAL_INDEX_ATTRIBUTE) is not None:
        raise ValueError("This mesh contains stale Mod Tool Base Color restore data.")
    attribute = mesh.attributes.new(_SP_ORIGINAL_INDEX_ATTRIBUTE, "INT", "FACE")
    attribute.data.foreach_set("value", values)


def _metadata(obj):
    encoded = obj.get(_SP_METADATA_KEY)
    if not encoded:
        raise ValueError("This object has no Mod Tool Base Color merge to restore.")
    try:
        return json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The stored Base Color merge metadata is invalid.") from exc


def _material_by_id(material_id):
    return next(
        (material for material in bpy.data.materials if material.get(_SP_MATERIAL_ID_KEY) == material_id),
        None,
    )


def _remap_active_uv(mesh, original_indices, mapping, grid_size, inverse=False):
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise ValueError(f"{mesh.name} has no active UV map.")
    uv_values = np.empty(len(uv_layer.data) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_values)
    loop_totals = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    polygon_indices = np.asarray(original_indices, dtype=np.int32)
    max_index = int(polygon_indices.max()) if polygon_indices.size else -1
    columns = np.zeros(max_index + 1, dtype=np.float32)
    rows = np.zeros(max_index + 1, dtype=np.float32)
    known = np.zeros(max_index + 1, dtype=bool)
    for material_index, entry in mapping.items():
        if material_index > max_index:
            continue
        tile = int(entry["tile"])
        columns[material_index] = tile % grid_size
        rows[material_index] = tile // grid_size
        known[material_index] = True
    if polygon_indices.size and not np.all(known[polygon_indices]):
        missing_face = int(np.flatnonzero(~known[polygon_indices])[0])
        raise ValueError(f"Face {missing_face} refers to an unknown original material slot.")
    loop_indices = np.repeat(polygon_indices, loop_totals)
    if len(loop_indices) != len(mesh.loops):
        raise RuntimeError("Polygon loop data changed while remapping the UV layer.")
    values = uv_values.reshape((-1, 2))
    loop_columns = columns[loop_indices]
    loop_rows = rows[loop_indices]
    if inverse:
        values[:, 0] = values[:, 0] * grid_size - loop_columns
        values[:, 1] = values[:, 1] * grid_size - loop_rows
    else:
        values[:, 0] = (values[:, 0] + loop_columns) / grid_size
        values[:, 1] = (values[:, 1] + loop_rows) / grid_size
    uv_layer.data.foreach_set("uv", uv_values)


def merge_base_color_for_sp(obj, directory, grid_size):
    """Make one SP material while storing exact metadata for a later restore."""
    _require_numpy()
    if obj is None or obj.type != "MESH":
        raise ValueError("Make one mesh object active.")
    if obj.get(_SP_METADATA_KEY):
        raise ValueError(f"{obj.name} is already prepared for Substance Painter. Restore it first.")
    if grid_size not in {2, 3, 4}:
        raise ValueError("Base Color grid size must be 2x2, 3x3, or 4x4.")
    mesh = obj.data
    if not mesh.uv_layers or mesh.uv_layers.active is None:
        raise ValueError(f"{obj.name} needs an active UV map.")
    source_slots = _collect_sources(obj)
    unique_sources = {}
    used_slots_by_pointer = {}
    for source in source_slots:
        pointer = source.material.as_pointer()
        unique_sources.setdefault(pointer, source)
        used_slots_by_pointer.setdefault(pointer, []).append(source.material_index)
    sources = list(unique_sources.values())
    busy_materials = [
        source.material.name
        for source in sources
        if source.material.get(_SP_MATERIAL_ID_KEY) is not None
        or source.material.get(_SP_PREVIOUS_FAKE_USER_KEY) is not None
    ]
    if busy_materials:
        raise ValueError(
            "These materials are already used by another pending Base Color merge: "
            + ", ".join(busy_materials)
        )
    if len(sources) > grid_size * grid_size:
        raise ValueError(f"{len(sources)} materials do not fit in a {grid_size}x{grid_size} grid.")
    directory = _resolve_directory(directory)
    tile_width = max(source.width for source in sources)
    tile_height = max(source.height for source in sources)
    atlas_width, atlas_height = tile_width * grid_size, tile_height * grid_size
    if atlas_width * atlas_height > _MAX_ATLAS_PIXELS:
        raise ValueError(
            f"{atlas_width}x{atlas_height} atlas is too large for in-memory Base Color processing. "
            "Use a smaller grid or source texture resolution."
        )

    original_indices = _read_polygon_material_indices(mesh)
    original_materials = list(mesh.materials)
    original_uv = np.empty(len(mesh.uv_layers.active.data) * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", original_uv)
    atlas_pixels = np.zeros((atlas_height, atlas_width, 4), dtype=np.float32)
    atlas_pixels[:, :, 3] = 1.0
    entries = []
    for tile, source in enumerate(sources):
        column, row = tile % grid_size, tile // grid_size
        left, bottom = column * tile_width, row * tile_height
        pixels = (
            _read_resampled_image_pixels(source.image, tile_width, tile_height)
            if source.image is not None
            else np.full((tile_height, tile_width, 4), source.color, dtype=np.float32)
        )
        atlas_pixels[bottom : bottom + tile_height, left : left + tile_width] = pixels
        material_id = uuid.uuid4().hex
        entries.append(
            {
                "material_indices": used_slots_by_pointer[source.material.as_pointer()],
                "material_id": material_id,
                "material_name": source.material.name,
                "tile": tile,
            }
        )

    atlas_stem = f"{_safe_name(obj.name)}_SP_BaseColor"
    atlas_path = os.path.join(directory, f"{atlas_stem}.png")
    atlas_image = None
    temporary_material = None
    tagged_materials = []
    try:
        atlas_image = _create_saved_image(atlas_stem, atlas_width, atlas_height, atlas_pixels, atlas_path)
        temporary_material = bpy.data.materials.new(name=f"{atlas_stem}_Material")
        _ensure_base_color_image_node(temporary_material, atlas_image, f"{atlas_stem}_Image")
        material_ids = {}
        for entry, source in zip(entries, sources):
            material = source.material
            material[_SP_MATERIAL_ID_KEY] = entry["material_id"]
            material[_SP_PREVIOUS_FAKE_USER_KEY] = bool(material.use_fake_user)
            material.use_fake_user = True
            tagged_materials.append(material)
            material_ids[material.as_pointer()] = entry["material_id"]
        slots = [
            {
                "material_id": material_ids[material.as_pointer()],
                "material_name": material.name,
            }
            for material in original_materials
        ]
        mapping = {
            material_index: entry
            for entry in entries
            for material_index in entry["material_indices"]
        }
        _ensure_original_index_attribute(mesh, original_indices)
        _remap_active_uv(
            mesh,
            original_indices,
            mapping,
            grid_size,
        )
        mesh.materials.clear()
        mesh.materials.append(temporary_material)
        mesh.polygons.foreach_set("material_index", array("i", [0]) * len(mesh.polygons))
        mesh.update()
        obj[_SP_METADATA_KEY] = json.dumps(
            {
                "version": 2,
                "grid_size": grid_size,
                "topology": _mesh_topology_signature(mesh),
                "uv_layer": mesh.uv_layers.active.name,
                "directory": directory,
                "atlas_path": atlas_path,
                "atlas_width": atlas_width,
                "atlas_height": atlas_height,
                "atlas_image": atlas_image.name,
                "temporary_material": temporary_material.name,
                "slots": slots,
                "entries": entries,
            },
            separators=(",", ":"),
        )
    except Exception:
        obj.pop(_SP_METADATA_KEY, None)
        attribute = mesh.attributes.get(_SP_ORIGINAL_INDEX_ATTRIBUTE)
        if attribute is not None:
            mesh.attributes.remove(attribute)
        mesh.uv_layers.active.data.foreach_set("uv", original_uv)
        mesh.materials.clear()
        for material in original_materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", original_indices)
        for material in tagged_materials:
            material.use_fake_user = bool(material.get(_SP_PREVIOUS_FAKE_USER_KEY, False))
            material.pop(_SP_MATERIAL_ID_KEY, None)
            material.pop(_SP_PREVIOUS_FAKE_USER_KEY, None)
        if temporary_material is not None and temporary_material.users == 0:
            bpy.data.materials.remove(temporary_material)
        if atlas_image is not None and atlas_image.users == 0:
            bpy.data.images.remove(atlas_image)
        mesh.update()
        raise
    return {
        "material_count": len(entries),
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
        "image_name": atlas_image.name,
        "filepath": atlas_path,
    }


def restore_base_color_from_sp(obj):
    """Split the painted atlas, restore materials, and reverse the UV transforms."""
    _require_numpy()
    if obj is None or obj.type != "MESH":
        raise ValueError("Make the SP-prepared mesh object active.")
    metadata = _metadata(obj)
    if metadata.get("version") != 2:
        raise ValueError("This Base Color merge uses the removed partial-tile workflow. Revert it, then merge again.")
    mesh = obj.data
    attribute = mesh.attributes.get(_SP_ORIGINAL_INDEX_ATTRIBUTE)
    if attribute is None or attribute.domain != "FACE":
        raise ValueError("The stored original material-index data is missing.")
    if len(attribute.data) != len(mesh.polygons):
        raise ValueError("Mesh topology changed after Base Color merge; it cannot be restored safely.")
    if _mesh_topology_signature(mesh) != metadata.get("topology"):
        raise ValueError("Mesh topology changed after Base Color merge; it cannot be restored safely.")
    uv_layer = mesh.uv_layers.get(metadata["uv_layer"])
    if uv_layer is None:
        raise ValueError(f"Stored UV layer {metadata['uv_layer']} no longer exists.")
    mesh.uv_layers.active = uv_layer
    atlas_image = bpy.data.images.get(metadata["atlas_image"])
    if atlas_image is None:
        atlas_image = bpy.data.images.load(metadata["atlas_path"], check_existing=True)
    if os.path.isfile(metadata["atlas_path"]):
        atlas_image.filepath = metadata["atlas_path"]
        atlas_image.reload()
    grid_size = int(metadata["grid_size"])
    if (
        int(atlas_image.size[0]) != int(metadata["atlas_width"])
        or int(atlas_image.size[1]) != int(metadata["atlas_height"])
    ):
        raise ValueError("The painted atlas dimensions changed; export it at its original resolution.")
    tile_width = int(atlas_image.size[0]) // grid_size
    tile_height = int(atlas_image.size[1]) // grid_size
    atlas_pixels = _read_image_pixels(atlas_image)
    original_indices = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", original_indices)
    entries = metadata["entries"]
    mapping = {
        int(material_index): entry
        for entry in entries
        for material_index in entry["material_indices"]
    }
    resolved_materials = {
        entry["material_id"]: _material_by_id(entry["material_id"])
        for entry in entries
    }
    missing = [
        entry["material_name"]
        for entry in entries
        if resolved_materials[entry["material_id"]] is None
    ]
    if missing:
        raise ValueError(f"Original materials are missing: {', '.join(missing)}")
    slot_materials = []
    for slot in metadata["slots"]:
        material = resolved_materials.get(slot["material_id"])
        if material is None:
            raise ValueError(f"Original material {slot['material_name']} is missing.")
        slot_materials.append(material)

    directory = _resolve_directory(metadata["directory"])
    prepared_images = []
    output_paths = []
    atlas_stem = os.path.splitext(os.path.basename(metadata["atlas_path"]))[0]
    try:
        for entry in entries:
            material = resolved_materials[entry["material_id"]]
            tile = int(entry["tile"])
            column, row = tile % grid_size, tile // grid_size
            if tile < 0 or tile >= grid_size * grid_size:
                raise ValueError("The stored Base Color tile metadata is invalid.")
            pixels = atlas_pixels[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ].copy()
            image_name = f"{atlas_stem}_{tile:02d}_{_safe_name(material.name)}"
            output_path = os.path.join(directory, f"{image_name}.png")
            image = _create_saved_image(image_name, tile_width, tile_height, pixels, output_path)
            prepared_images.append((entry, material, image, image_name))
            output_paths.append(output_path)
    except Exception:
        for _entry, _material, image, _image_name in prepared_images:
            if image.users == 0:
                bpy.data.images.remove(image)
        raise

    prepared_uv = np.empty(len(uv_layer.data) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", prepared_uv)
    prepared_materials = list(mesh.materials)
    prepared_indices = _read_polygon_material_indices(mesh)
    node_snapshots = []
    try:
        _remap_active_uv(
            mesh,
            original_indices,
            mapping,
            grid_size,
            inverse=True,
        )
        for _entry, material, image, image_name in prepared_images:
            previous_node = _base_color_image_node(material)
            previous_image = previous_node.image if previous_node is not None else None
            image_node = _ensure_base_color_image_node(material, image, f"{image_name}_BaseColor")
            node_snapshots.append((material, image_node, previous_image, previous_node is None))
        mesh.materials.clear()
        for material in slot_materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", original_indices)
        mesh.update()
    except Exception:
        uv_layer.data.foreach_set("uv", prepared_uv)
        mesh.materials.clear()
        for material in prepared_materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", prepared_indices)
        for material, image_node, previous_image, created_node in reversed(node_snapshots):
            if created_node and image_node.id_data == material.node_tree:
                material.node_tree.nodes.remove(image_node)
            else:
                image_node.image = previous_image
        for _entry, _material, image, _image_name in prepared_images:
            if image.users == 0:
                bpy.data.images.remove(image)
        mesh.update()
        raise

    mesh.attributes.remove(attribute)
    for entry in entries:
        material = resolved_materials[entry["material_id"]]
        material.use_fake_user = bool(material.get(_SP_PREVIOUS_FAKE_USER_KEY, False))
        material.pop(_SP_MATERIAL_ID_KEY, None)
        material.pop(_SP_PREVIOUS_FAKE_USER_KEY, None)
    temporary_material = bpy.data.materials.get(metadata["temporary_material"])
    if temporary_material is not None and temporary_material.users == 0:
        bpy.data.materials.remove(temporary_material)
    if atlas_image.users == 0:
        bpy.data.images.remove(atlas_image)
    obj.pop(_SP_METADATA_KEY, None)
    mesh.update()
    return {"material_count": len(prepared_images), "output_paths": output_paths}
