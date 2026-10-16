"""Bake output-connected color chains using independent sampling and destination UVs."""

from __future__ import annotations

import os
from collections import defaultdict

import bpy

from .materials import _MAX_ATLAS_PIXELS, _read_polygon_material_indices, _require_numpy, _safe_name, np


def prepare_bake_materials(obj, settings):
    """Explicitly refresh the small UI cache; never inspect mesh faces from draw()."""
    if not obj.data.uv_layers.active:
        raise ValueError("The active mesh needs a source UV map.")
    settings.mesh_object = obj
    settings.source_uv = obj.data.uv_layers.active.name
    settings.materials.clear()
    for index in sorted(set(_read_polygon_material_indices(obj.data))):
        if index >= len(obj.material_slots) or obj.material_slots[index].material is None:
            raise ValueError(f"Material slot {index} is empty.")
        row = settings.materials.add()
        row.slot_index = index
        row.material = obj.material_slots[index].material
        row.target_uv = settings.source_uv
    if not settings.materials:
        raise ValueError("The active mesh has no material faces to bake.")


def _destination_uv(mesh, targets):
    """Read each requested UV once and assemble per-material face corners in NumPy."""
    material_indices = np.asarray(_read_polygon_material_indices(mesh), dtype=np.int32)
    if not len(material_indices) or set(material_indices) != set(targets):
        raise ValueError("Used materials changed. Read Materials again and check the destination UVs.")
    counts = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", counts)
    loop_materials = np.repeat(material_indices, counts)
    destination = np.empty((len(mesh.loops), 2), dtype=np.float32)
    for name in set(targets.values()):
        layer = mesh.uv_layers.get(name)
        if layer is None:
            raise ValueError(f"Destination UV map '{name}' does not exist.")
        values = np.empty_like(destination)
        layer.data.foreach_get("uv", values.ravel())
        mask = np.isin(loop_materials, [i for i, uv in targets.items() if uv == name])
        destination[mask] = values[mask]
    if not np.isfinite(destination).all():
        raise ValueError("Destination UVs contain invalid coordinates.")
    if np.any(destination < -1e-6) or np.any(destination > 1.000001):
        raise ValueError("Destination UVs must fit inside the 0–1 tile for a single image.")
    return destination, material_indices


def _check_destination(mesh, uv, material_indices, resolution):
    """Reject empty UV layouts and cross-material collisions at bake pixel centers."""
    mesh.calc_loop_triangles()
    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    faces = np.empty(len(triangles), dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", triangles.ravel())
    mesh.loop_triangles.foreach_get("polygon_index", faces)
    positions = uv[triangles].astype(np.float64)
    edges_a = positions[:, 1] - positions[:, 0]
    edges_b = positions[:, 2] - positions[:, 0]
    area = edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0]
    materials = material_indices[faces]
    for index in set(material_indices):
        if not np.any(np.abs(area[materials == index]) > 1e-12):
            raise ValueError(f"Material slot {index} has no nonzero-area destination UV faces.")
    # A single material may deliberately reuse its own UV islands. Different materials
    # cannot write different colors into the same output pixels.
    if len(set(material_indices)) > 1:
        owners = np.full((resolution, resolution), -1, dtype=np.int32)
        for tri, signed_area, material in zip(positions, area, materials):
            if abs(signed_area) < 1e-12:
                continue
            low = np.maximum(0, np.ceil(tri.min(axis=0) * resolution - .5).astype(int))
            high = np.minimum(resolution - 1, np.floor(tri.max(axis=0) * resolution - .5).astype(int))
            if np.any(high < low):
                continue
            x = (np.arange(low[0], high[0] + 1) + .5) / resolution - tri[0, 0]
            y = (np.arange(low[1], high[1] + 1) + .5) / resolution - tri[0, 1]
            a, b = tri[1] - tri[0], tri[2] - tri[0]
            u = (x[None, :] * b[1] - y[:, None] * b[0]) / signed_area
            v = (a[0] * y[:, None] - a[1] * x[None, :]) / signed_area
            inside = (u > 1e-8) & (v > 1e-8) & (u + v < 1 - 1e-8)
            region = owners[low[1]:high[1] + 1, low[0]:high[0] + 1]
            conflict = inside & (region >= 0) & (region != material)
            if np.any(conflict):
                other = int(region[conflict][0])
                raise ValueError(f"Destination UVs overlap between material slots {other} and {material}. Move the destination islands apart.")
            region[inside] = material
    return int(np.count_nonzero(np.abs(area) < 1e-12))


def _prune_and_bind(tree, roots, source_uv, owned_groups, ancestors=(), keep=()):
    """Keep the color dependency graph and make implicit texture UVs explicit."""
    seen = set(keep)
    group_outputs = defaultdict(set)
    pending = list(roots)
    while pending:
        socket = pending.pop()
        node = socket.node
        if node.type == "GROUP":
            group_outputs[node.name].add(list(node.outputs).index(socket))
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "SHADER_TO_RGB":
            raise ValueError("The connected color chain contains Shader to RGB (EEVEE only). Connect the color chain before lighting instead.")
        if node.type == "TEX_IMAGE" and node.image is None:
            raise ValueError(f"Connected image texture '{node.name}' has no image.")
        pending.extend(link.from_socket for entry in node.inputs for link in entry.links)
    for node in list(tree.nodes):
        if node.name not in seen:
            tree.nodes.remove(node)
    for node in list(tree.nodes):
        if node.type == "GROUP" and node.node_tree:
            original = node.node_tree
            pointer = original.as_pointer()
            if pointer in ancestors:
                raise ValueError("Recursive shader groups cannot be baked.")
            copied = original.copy()
            owned_groups.append(copied)
            node.node_tree = copied
            output = next((n for n in copied.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None)
            if output is None:
                raise ValueError(f"Shader group '{original.name}' has no active output.")
            required = group_outputs[node.name]
            group_roots = [l.from_socket for i in required for l in output.inputs[i].links]
            # Keep the group output itself while pruning its unrelated branches.
            for i, entry in enumerate(output.inputs):
                if i not in required:
                    for link in list(entry.links):
                        copied.links.remove(link)
            _prune_and_bind(copied, group_roots, source_uv, owned_groups, ancestors + (pointer,), keep=(output.name,))
        elif node.type == "UVMAP" and not node.uv_map:
            node.uv_map = source_uv
        elif node.type == "TEX_IMAGE" and not node.inputs["Vector"].is_linked:
            uv_node = tree.nodes.new("ShaderNodeUVMap")
            uv_node.uv_map = source_uv
            tree.links.new(uv_node.outputs["UV"], node.inputs["Vector"])
        elif node.type == "TEX_COORD":
            links = list(node.outputs["UV"].links)
            if links:
                uv_node = tree.nodes.new("ShaderNodeUVMap")
                uv_node.uv_map = source_uv
                for link in links:
                    tree.links.new(uv_node.outputs["UV"], link.to_socket)


def _color_material(material, source_uv, image, engine, owned_materials, owned_groups):
    if material is None or material.node_tree is None:
        raise ValueError("Every used material needs a color chain connected to Material Output.")
    output = material.node_tree.get_output_node("EEVEE" if engine == "BLENDER_EEVEE" else "CYCLES")
    links = output.inputs["Surface"].links if output else ()
    if len(links) != 1 or links[0].from_socket.type not in {"RGBA", "VALUE", "VECTOR"}:
        raise ValueError(f"{material.name}: connect the desired Color/Value output directly to Material Output Surface before baking.")
    link = links[0]
    socket_index = list(link.from_node.outputs).index(link.from_socket)
    copied = material.copy()
    owned_materials.append(copied)
    tree = copied.node_tree
    color = tree.nodes[link.from_node.name].outputs[socket_index]
    _prune_and_bind(tree, [color], source_uv, owned_groups)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1
    tree.links.new(color, emission.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    target = tree.nodes.new("ShaderNodeTexImage")
    target.image = image
    for node in tree.nodes:
        node.select = node == target
    tree.nodes.active = target
    return copied


def bake_output_colors(context, obj, settings):
    """Bake in an isolated scene, leaving the original geometry and shaders untouched."""
    _require_numpy("Output color bake")
    if settings.mesh_object != obj or not settings.materials:
        raise ValueError("Click Read Materials on the active mesh first.")
    if obj.data.uv_layers.get(settings.source_uv) is None:
        raise ValueError(f"Source UV '{settings.source_uv}' does not exist.")
    targets = {}
    for row in settings.materials:
        if row.slot_index >= len(obj.material_slots) or obj.material_slots[row.slot_index].material != row.material:
            raise ValueError("Material slots changed. Read Materials again.")
        targets[row.slot_index] = row.target_uv
    original_uv, _ = _destination_uv(obj.data, targets)
    resolution = int(settings.resolution)
    if resolution * resolution > _MAX_ATLAS_PIXELS:
        raise ValueError("Bake image exceeds the supported pixel limit.")
    filepath = os.path.abspath(bpy.path.abspath(settings.filepath))
    if not settings.filepath or os.path.splitext(filepath)[1].lower() != ".png":
        raise ValueError("Choose a PNG output file.")
    if os.path.exists(filepath):
        raise ValueError(f"Output file already exists. Choose a new filename: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mesh = None
    scene = None
    temporary = None
    image = None
    owned_materials = []
    owned_groups = []
    success = False
    try:
        depsgraph = context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
        uv, indices = _destination_uv(mesh, targets)
        degenerate = _check_destination(mesh, uv, indices, resolution)
        if mesh.uv_layers.get(settings.source_uv) is None:
            raise ValueError("Evaluated mesh lost the source UV map.")
        destination = mesh.uv_layers.new(name=".mdt_bake_destination")
        destination.data.foreach_set("uv", uv.ravel())
        destination.active_render = True
        mesh.uv_layers.active_index = len(mesh.uv_layers) - 1
        image = bpy.data.images.new(f"{_safe_name(obj.name)}_OutputColor", width=resolution, height=resolution, alpha=True, float_buffer=True)
        image.colorspace_settings.name = "sRGB"
        image.generated_color = (0, 0, 0, 0)
        mesh.materials.clear()
        for slot in range(max(targets) + 1):
            material = _color_material(obj.material_slots[slot].material, settings.source_uv, image, context.scene.render.engine, owned_materials, owned_groups) if slot in targets else None
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", indices)
        temporary = bpy.data.objects.new(".MDT_ColorBake", mesh)
        temporary.matrix_world = obj.matrix_world.copy()
        scene = bpy.data.scenes.new(".MDT_ColorBake")
        scene.collection.objects.link(temporary)
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 4
        scene.render.bake.use_selected_to_active = False
        layer = scene.view_layers[0]
        layer.objects.active = temporary
        temporary.select_set(True, view_layer=layer)
        with context.temp_override(scene=scene, view_layer=layer, object=temporary, active_object=temporary, selected_objects=[temporary], selected_editable_objects=[temporary]):
            result = bpy.ops.object.bake(type="EMIT", uv_layer=destination.name, target="IMAGE_TEXTURES", use_clear=True, use_selected_to_active=False, margin=settings.margin, margin_type="EXTEND")
        if "FINISHED" not in result:
            raise RuntimeError("Color bake was cancelled.")
        image.filepath_raw = filepath
        image.file_format = "PNG"
        image.save()
        # A distinct UV layer makes the per-material composite usable with one image.
        # Existing UV maps and their active/render selection remain unchanged.
        active_index = obj.data.uv_layers.active_index
        render_uv = next((u.name for u in obj.data.uv_layers if u.active_render), None)
        output_uv = obj.data.uv_layers.new(name="BakedUV")
        output_uv.data.foreach_set("uv", original_uv.ravel())
        obj.data.uv_layers.active_index = active_index
        if render_uv:
            obj.data.uv_layers[render_uv].active_render = True
        image.use_fake_user = True
        settings.last_image = image
        settings.last_uv = output_uv.name
        success = True
        return {"image": image, "filepath": filepath, "uv_name": output_uv.name, "material_count": len(targets), "degenerate_triangles": degenerate}
    finally:
        if temporary:
            bpy.data.objects.remove(temporary, do_unlink=True)
        if scene:
            bpy.data.scenes.remove(scene)
        if mesh:
            bpy.data.meshes.remove(mesh)
        for material in owned_materials:
            bpy.data.materials.remove(material)
        for group in reversed(owned_groups):
            bpy.data.node_groups.remove(group)
        if image and not success:
            bpy.data.images.remove(image)
