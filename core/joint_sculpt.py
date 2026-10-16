"""Fixed-topology native sculpting on a temporary union of independent meshes."""

from __future__ import annotations

import hashlib
import json

import bpy

from .common import get_flat_point_coordinates, mesh_topology_signature, restore_context_mode
from .materials import _require_numpy, np


_START = ".mdt_joint_sculpt_start"
_IDS = ".mdt_joint_sculpt_id"
_RENDER_UV = ".mdt_joint_sculpt_render"
_AUTOMASK = ("use_automasking_topology", "use_automasking_face_sets")


def _read(elements, field, width=1, dtype="int32"):
    values = np.empty(len(elements) * width, dtype=dtype)
    if len(values):
        elements.foreach_get(field, values)
    return values.reshape((-1, width)) if width > 1 else values


def _coordinates(points):
    return np.frombuffer(get_flat_point_coordinates(points), dtype=np.float32).reshape((-1, 3))


def _geometry_snapshot(obj):
    """Collect once for validation and later writes/rollback, including every key."""
    mesh = obj.data
    coordinates = _coordinates(mesh.vertices)
    keys = [(key, _coordinates(key.data)) for key in mesh.shape_keys.key_blocks] if mesh.shape_keys else []
    digest = hashlib.sha256()
    topology = mesh_topology_signature(mesh, include_edges=True)
    digest.update(topology.encode())
    digest.update(coordinates.tobytes())
    digest.update(np.asarray(obj.matrix_world, dtype=np.float64).tobytes())
    digest.update(repr((obj.active_shape_key_index, obj.show_only_shape_key)).encode())
    for modifier in obj.modifiers:
        state = (modifier.name, modifier.type, modifier.show_viewport)
        if modifier.type == "ARMATURE":
            state += (modifier.object.name if modifier.object else None,
                      modifier.use_vertex_groups, modifier.use_bone_envelopes,
                      modifier.use_deform_preserve_volume, modifier.use_multi_modifier,
                      modifier.vertex_group, modifier.invert_vertex_group)
        digest.update(repr(state).encode())
    if mesh.shape_keys:
        digest.update(repr((mesh.shape_keys.use_relative, mesh.shape_keys.eval_time)).encode())
    for key, values in keys:
        digest.update(values.tobytes())
        digest.update(repr((key.name, key.value, key.mute, key.lock_shape, key.relative_key.name if key.relative_key else None, key.vertex_group, key.interpolation)).encode())
    return coordinates, keys, digest.hexdigest(), topology


def _check_modifiers(obj):
    armature_enabled = False
    for modifier in obj.modifiers:
        if not modifier.show_viewport:
            continue
        if modifier.type != "ARMATURE":
            raise ValueError(f"{obj.name}: disable '{modifier.name}' before Joint Sculpt; this tool requires unchanged topology and supports Armature modifiers.")
        armature_enabled |= modifier.object is not None
    return armature_enabled


def _evaluated_coordinates(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError(f"{obj.name}: evaluated vertex correspondence changed.")
        return _coordinates(mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def _original_displacements(context, plans):
    """Measure the actual skinning map on disposable, shape-mixed meshes.

    Native crazyspace matrices omit part of an Armature influence mask and
    include the active shape key's editing factor. Three bulk coordinate probes
    include the real modifier stack, masks, bone scale and Preserve Volume,
    without per-vertex RNA calls or changes to the originals. Nonlinear skinning
    is checked again against the actual sculpt before the operation completes.
    """
    results = {}
    probes = []
    meshes = []
    try:
        for obj, _coordinates_before, _keys, delta, visible in plans:
            if visible is None or not np.any(delta):
                continue
            probe = obj.copy()
            probes.append((obj, probe, delta, visible))
            probe.name = "JointSculpt_DeformProbe"
            probe.animation_data_clear()
            probe.constraints.clear()
            probe.parent = None
            probe.matrix_world = obj.matrix_world
            probe.hide_render = True
            context.collection.objects.link(probe)
            probe.hide_set(False)
            for modifier in probe.modifiers:
                modifier.show_viewport = False
        if not probes:
            return results
        # First ask Blender for the current shape-key mix, retaining sparse weights.
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        samples = []
        for obj, probe, delta, visible in probes:
            part = bpy.data.meshes.new_from_object(probe.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
            meshes.append(part)
            probe.data = part
            for source, modifier in zip(obj.modifiers, probe.modifiers):
                modifier.show_viewport = source.show_viewport
            positions = _coordinates(part.vertices)
            indices = np.flatnonzero(np.any(delta != 0, axis=1))
            step = max(float(np.ptp(positions, axis=0).max()) * .005,
                       float(np.abs(positions).max()) * 1e-5, 1e-3)
            samples.append((obj, probe, delta, visible, positions, indices, step,
                            np.empty((len(indices), 3, 3), dtype=np.float64)))
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        for obj, probe, _delta, visible, _positions, _indices, _step, _mapping in samples:
            if not np.allclose(_evaluated_coordinates(probe, depsgraph), visible, atol=2e-6, rtol=1e-6):
                raise ValueError(f"{obj.name}: could not reproduce the starting bone deformation for inverse sculpting.")
        for axis in range(3):
            for _obj, probe, _delta, _visible, positions, indices, step, _mapping in samples:
                displaced = positions.copy()
                displaced[indices, axis] += step
                probe.data.vertices.foreach_set("co", displaced.ravel())
                probe.data.update()
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            for _obj, probe, _delta, visible, _positions, indices, step, mapping in samples:
                mapping[:, :, axis] = (_evaluated_coordinates(probe, depsgraph)[indices].astype(np.float64)
                                       - visible[indices]) / step
        for obj, _probe, delta, _visible, _positions, indices, _step, mapping in samples:
            if not np.isfinite(mapping).all():
                raise ValueError(f"{obj.name}: invalid bone deformation map.")
            singular_values = np.linalg.svd(mapping, compute_uv=False)
            if np.any(singular_values[:, -1] < np.maximum(1e-8, singular_values[:, 0] * 1e-6)):
                raise ValueError(f"{obj.name}: bone deformation cannot invert this sculpt displacement (check zero bone scale).")
            local_delta = np.zeros_like(delta)
            local_delta[indices] = np.linalg.solve(mapping, delta[indices, :, None])[..., 0]
            if not np.isfinite(local_delta).all():
                raise ValueError(f"{obj.name}: invalid inverse bone displacement.")
            results[obj] = local_delta
        return results
    finally:
        for _obj, probe, _delta, _visible in probes:
            bpy.data.objects.remove(probe, do_unlink=True)
        for mesh in meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _build_proxy(parts, frame_inverse):
    mesh = bpy.data.meshes.new("JointSculpt_Mesh")
    try:
        uv_names = sorted({layer.name for obj, part in parts for layer in part.uv_layers})
        if len(uv_names) >= 8:
            raise ValueError("Joint Sculpt needs one free UV slot for the combined render UV.")
        vertices, edges, loops, starts, totals, smooth, material_indices = [], [], [], [], [], [], []
        masks, render_uvs = [], []
        uv_chunks = {name: [] for name in uv_names}
        materials, material_map = [], {}
        offsets = []
        vertex_offset = loop_offset = 0
        for obj, part in parts:
            offsets.append((vertex_offset, len(part.vertices)))
            transform = frame_inverse @ np.asarray(obj.matrix_world, dtype=np.float64)
            values = _coordinates(part.vertices)
            vertices.append((values @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32))
            edges.append(_read(part.edges, "vertices", 2) + vertex_offset)
            face_starts = _read(part.polygons, "loop_start")
            face_totals = _read(part.polygons, "loop_total")
            order = np.arange(len(part.loops))
            if np.linalg.det(transform[:3, :3]) < 0:
                order = np.repeat(2 * face_starts + face_totals - 1, face_totals) - order
            loops.append(_read(part.loops, "vertex_index")[order] + vertex_offset)
            starts.append(face_starts + loop_offset)
            totals.append(face_totals)
            smooth.append(_read(part.polygons, "use_smooth", dtype=np.bool_))
            slot_map = []
            for slot in obj.material_slots:
                material = slot.material
                pointer = material.as_pointer() if material else 0
                if pointer not in material_map:
                    material_map[pointer] = len(materials)
                    materials.append(material)
                slot_map.append(material_map[pointer])
            if not slot_map:
                if 0 not in material_map:
                    material_map[0] = len(materials)
                    materials.append(None)
                slot_map = [material_map[0]]
            indices = _read(part.polygons, "material_index")
            if np.any(indices >= len(slot_map)):
                raise ValueError(f"{obj.name}: faces refer to missing material slots.")
            material_indices.append(np.asarray(slot_map, dtype=np.int32)[indices])
            for name in uv_names:
                layer = part.uv_layers.get(name)
                uv_chunks[name].append(_read(layer.data, "uv", 2, np.float32)[order] if layer else np.zeros((len(part.loops), 2), dtype=np.float32))
            render = next((u for u in part.uv_layers if u.active_render), part.uv_layers.active)
            render_uvs.append(_read(render.data, "uv", 2, np.float32)[order] if render else np.zeros((len(part.loops), 2), dtype=np.float32))
            mask = part.attributes.get(".sculpt_mask")
            masks.append(_read(mask.data, "value", dtype=np.float32) if mask else np.zeros(len(part.vertices), dtype=np.float32))
            vertex_offset += len(part.vertices)
            loop_offset += len(part.loops)
        positions = np.concatenate(vertices)
        mesh.vertices.add(len(positions))
        mesh.vertices.foreach_set("co", positions.ravel())
        mesh.edges.add(sum(len(a) for a in edges))
        mesh.edges.foreach_set("vertices", np.concatenate(edges).ravel())
        mesh.loops.add(loop_offset)
        mesh.loops.foreach_set("vertex_index", np.concatenate(loops))
        mesh.polygons.add(sum(len(a) for a in starts))
        mesh.polygons.foreach_set("loop_start", np.concatenate(starts))
        mesh.polygons.foreach_set("loop_total", np.concatenate(totals))
        mesh.polygons.foreach_set("use_smooth", np.concatenate(smooth))
        for material in materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", np.concatenate(material_indices))
        for name, chunks in uv_chunks.items():
            layer = mesh.uv_layers.new(name=name)
            layer.data.foreach_set("uv", np.concatenate(chunks).ravel())
        if uv_names:
            layer = mesh.uv_layers.new(name=_RENDER_UV)
            layer.data.foreach_set("uv", np.concatenate(render_uvs).ravel())
            layer.active_render = True
        mesh.attributes.new(_START, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", positions.ravel())
        mesh.attributes.new(_IDS, "INT", "POINT").data.foreach_set("value", np.arange(vertex_offset, dtype=np.int32))
        if any(np.any(mask) for mask in masks):
            mesh.attributes.new(".sculpt_mask", "FLOAT", "POINT").data.foreach_set("value", np.concatenate(masks))
        mesh.update(calc_edges=True)
        return mesh, offsets
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise


def _set_automasking(context, settings):
    settings.brush = context.scene.tool_settings.sculpt.brush
    backup = {}
    for name, owner in (("scene", context.scene.tool_settings.sculpt), ("brush", settings.brush)):
        if owner:
            backup[name] = {field: getattr(owner, field) for field in _AUTOMASK if hasattr(owner, field)}
            for field in backup[name]:
                setattr(owner, field, False)
    settings.automasking_backup = json.dumps(backup)


def _restore_session(context, settings):
    """Restore visibility, selection and mode; also works if the proxy was deleted."""
    layer = context.scene.view_layers.get(settings.view_layer_name) or context.view_layer
    previous_mode, previous_active = settings.previous_mode, settings.previous_active
    selected = [item.object for item in settings.selection if item.object]
    for row in settings.sources:
        if row.object and layer.objects.get(row.object.name):
            row.object.hide_set(row.hidden, view_layer=layer)
    backup = json.loads(settings.automasking_backup or "{}")
    for name, owner in (("scene", context.scene.tool_settings.sculpt), ("brush", settings.brush)):
        if owner:
            for field, value in backup.get(name, {}).items():
                setattr(owner, field, value)
    proxy, mesh = settings.proxy, settings.proxy_mesh
    settings.proxy = None
    settings.proxy_mesh = None
    if proxy:
        bpy.data.objects.remove(proxy, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    for obj in list(context.selected_objects):
        obj.select_set(False)
    for obj in selected:
        if context.view_layer.objects.get(obj.name) and obj.visible_get(view_layer=context.view_layer):
            obj.select_set(True)
    if previous_active and context.view_layer.objects.get(previous_active.name):
        context.view_layer.objects.active = previous_active
    settings.sources.clear()
    settings.selection.clear()
    settings.previous_active = None
    settings.previous_mode = ""
    settings.view_layer_name = ""
    settings.topology_hash = ""
    settings.brush = None
    settings.automasking_backup = ""
    restore_context_mode(previous_mode)


def start_joint_sculpt(context, objects, previous_mode):
    _require_numpy("Joint Sculpt")
    settings = context.scene.mdt_joint_sculpt_settings
    if settings.sources or settings.proxy:
        raise ValueError("Apply or cancel the existing Joint Sculpt session first.")
    if len(objects) < 2:
        raise ValueError("Select at least two mesh objects for Joint Sculpt.")
    if context.object not in objects:
        raise ValueError("Make one of the selected mesh objects active.")
    snapshots = []
    for obj in objects:
        if not obj.is_editable or not obj.data.is_editable:
            raise ValueError(f"{obj.name}: the mesh is not editable.")
        if obj.data.users - int(obj.data.use_fake_user) > 1:
            raise ValueError(f"{obj.name}: make the mesh data single-user before Joint Sculpt.")
        if not len(obj.data.vertices) or not len(obj.data.polygons):
            raise ValueError(f"{obj.name}: Joint Sculpt requires a surface mesh.")
        if abs(np.linalg.det(np.asarray(obj.matrix_world)[:3, :3])) < 1e-12:
            raise ValueError(f"{obj.name}: zero scale cannot be mapped back.")
        _check_modifiers(obj)
        snapshot = _geometry_snapshot(obj)
        snapshots.append((snapshot[2], snapshot[3]))
    location, rotation, _scale = context.object.matrix_world.decompose()
    frame = rotation.to_matrix().to_4x4()
    frame.translation = location
    inverse = np.linalg.inv(np.asarray(frame, dtype=np.float64))
    evaluated_parts = []
    mesh = None
    try:
        depsgraph = context.evaluated_depsgraph_get()
        for obj, (_state_hash, topology) in zip(objects, snapshots):
            part = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
            evaluated_parts.append((obj, part))
            if mesh_topology_signature(part, include_edges=True) != topology:
                raise ValueError(f"{obj.name}: evaluated topology differs from the original mesh.")
        mesh, offsets = _build_proxy(evaluated_parts, inverse)
    finally:
        for obj, part in evaluated_parts:
            bpy.data.meshes.remove(part)
    settings.previous_active = context.object
    settings.previous_mode = previous_mode
    settings.view_layer_name = context.view_layer.name
    for obj in context.selected_objects:
        settings.selection.add().object = obj
    settings.proxy_mesh = mesh
    try:
        proxy = bpy.data.objects.new("JointSculpt", mesh)
        settings.proxy = proxy
        proxy.matrix_world = frame
        proxy.hide_render = True
        context.collection.objects.link(proxy)
        space = context.space_data
        if space and space.type == "VIEW_3D" and space.local_view:
            proxy.local_view_set(space, True)
        for obj, (start, count), (state_hash, _topology) in zip(objects, offsets, snapshots):
            row = settings.sources.add()
            row.object, row.mesh = obj, obj.data
            row.mesh_users = obj.data.users
            row.start, row.count, row.state_hash = start, count, state_hash
            row.hidden = obj.hide_get()
            obj.hide_set(True)
        for obj in list(context.selected_objects):
            obj.select_set(False)
        proxy.select_set(True)
        context.view_layer.objects.active = proxy
        settings.topology_hash = mesh_topology_signature(mesh, include_edges=True)
        settings.proxy_matrix = np.asarray(frame, dtype=np.float64).ravel()
        _set_automasking(context, settings)
        bpy.ops.object.mode_set(mode="SCULPT")
        return {"object_count": len(objects), "vertex_count": len(mesh.vertices), "proxy": proxy}
    except Exception:
        _restore_session(context, settings)
        raise


def apply_joint_sculpt(context):
    _require_numpy("Joint Sculpt")
    settings = context.scene.mdt_joint_sculpt_settings
    proxy = settings.proxy
    if not settings.sources or not proxy or proxy.data != settings.proxy_mesh:
        raise ValueError("The Joint Sculpt proxy is missing. Cancel the session to restore the original objects.")
    if context.view_layer.name != settings.view_layer_name:
        raise ValueError("Return to the view layer where Joint Sculpt started before applying.")
    mesh = proxy.data
    if proxy.use_dynamic_topology_sculpting or mesh_topology_signature(mesh, include_edges=True) != settings.topology_hash:
        raise ValueError("Joint Sculpt topology changed. Undo the topology change or cancel; original objects have not been modified.")
    ids, start = mesh.attributes.get(_IDS), mesh.attributes.get(_START)
    if ids is None or start is None or not np.array_equal(_read(ids.data, "value"), np.arange(len(mesh.vertices))):
        raise ValueError("Joint Sculpt vertex correspondence was changed; original objects have not been modified.")
    matrix = np.asarray(proxy.matrix_world, dtype=np.float64)
    if not np.allclose(matrix.ravel(), settings.proxy_matrix, atol=1e-7, rtol=0):
        raise ValueError("The proxy object transform changed. Restore its transform before applying vertex edits.")
    start_positions = _read(start.data, "vector", 3, np.float32)
    delta = _coordinates(mesh.vertices) - start_positions
    if not np.isfinite(delta).all():
        raise ValueError("Joint Sculpt contains invalid vertex coordinates.")
    plans = []
    depsgraph = context.evaluated_depsgraph_get()
    for row in settings.sources:
        obj = row.object
        if not obj or obj.data != row.mesh or obj.data.users != row.mesh_users:
            raise ValueError("An original object or its mesh data changed. Cancel Joint Sculpt; nothing has been written back.")
        armature_enabled = _check_modifiers(obj)
        coordinates, keys, state_hash, _topology = _geometry_snapshot(obj)
        if state_hash != row.state_hash:
            raise ValueError(f"{obj.name}: original geometry, transforms or shape-key state changed during Joint Sculpt.")
        transform = np.linalg.inv(np.asarray(obj.matrix_world, dtype=np.float64))[:3, :3] @ matrix[:3, :3]
        evaluated_delta = (delta[row.start:row.start + row.count] @ transform.T).astype(np.float32)
        visible = None
        if armature_enabled:
            visible = _evaluated_coordinates(obj, depsgraph)
            to_proxy = np.linalg.inv(matrix) @ np.asarray(obj.matrix_world, dtype=np.float64)
            current = visible @ to_proxy[:3, :3].T + to_proxy[:3, 3]
            if not np.allclose(current, start_positions[row.start:row.start + row.count], atol=2e-6, rtol=1e-6):
                raise ValueError(f"{obj.name}: bone deformation changed during Joint Sculpt. Restore the starting pose or cancel.")
        plans.append((obj, coordinates, keys, evaluated_delta, visible))
    # Build all write plans before changing any geometry. Preserve key values,
    # relative deltas and the modifier stack; do not bake the posed mesh into Basis.
    writes = []
    displacements = _original_displacements(context, plans)
    for obj, coordinates, keys, evaluated_delta, visible in plans:
        local_delta = displacements.get(obj, evaluated_delta)
        targets = []
        if np.any(local_delta):
            targets.append((obj.data.vertices, coordinates, coordinates + local_delta))
            targets.extend((key.data, values, values + local_delta) for key, values in keys)
        writes.append((obj, targets, visible + evaluated_delta if visible is not None else None))
    # Validate every object before writing any. Retain old arrays for rollback if an
    # RNA write fails, so applying cannot leave only some objects updated.
    try:
        for obj, targets, _expected in writes:
            for points, old, new in targets:
                points.foreach_set("co", new.ravel())
            if targets:
                obj.data.update()
        if any(targets and expected is not None for _obj, targets, expected in writes):
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            for obj, targets, expected in writes:
                if targets and expected is not None:
                    actual = _evaluated_coordinates(obj, depsgraph)
                    if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=3e-6, rtol=1e-6):
                        raise ValueError(f"{obj.name}: bone deformation could not reproduce the sculpt. Original geometry was restored; the sculpt session is retained.")
    except Exception:
        for obj, targets, _expected in writes:
            for points, old, new in targets:
                points.foreach_set("co", old.ravel())
            if targets:
                obj.data.update()
        context.view_layer.update()
        raise
    result = {"object_count": len(plans), "changed_vertices": int(np.count_nonzero(np.any(delta != 0, axis=1)))}
    _restore_session(context, settings)
    return result


def cancel_joint_sculpt(context):
    settings = context.scene.mdt_joint_sculpt_settings
    if not settings.sources and not settings.proxy:
        raise ValueError("No Joint Sculpt session is active.")
    _restore_session(context, settings)
