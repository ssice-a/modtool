"""Editable Shrinkwrap proxies driving one smooth, native lattice deformation."""

from __future__ import annotations

import itertools
import json
import uuid

import bpy
from mathutils import Matrix

from .common import mesh_coordinate_offset_plan, mesh_topology_signature, run_object_operator
from .materials import _require_numpy, np
from .modifier_apply import _flat_vertex_group_weights
from .rename import _mesh_triangle_array, _mirror_coordinate_array


_REST = ".mdt_fit_rest"
_IDS = ".mdt_fit_id"
_OWNER = "mdt_proxy_fit_owner"


def _world(obj, co):
    matrix = np.asarray(obj.matrix_world)
    return co @ matrix[:3, :3].T + matrix[:3, 3]


def _coordinates(obj, evaluated=False):
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()) if evaluated else None
    mesh = ev.to_mesh() if ev else obj.data
    try:
        co = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get("co", co.ravel())
        return _world(obj, co)
    finally:
        if ev:
            ev.to_mesh_clear()


def _make_proxy(context, obj, face_budget, owner, role):
    co = _world(obj, _mirror_coordinate_array(obj))
    tri = _mesh_triangle_array(obj.data)
    if obj.matrix_world.to_3x3().determinant() < 0:
        tri = tri[:, ::-1].copy()
    if not len(co) or not len(tri) or not np.isfinite(co).all():
        raise ValueError(f"{obj.name}: choose a finite surface mesh.")
    mesh = bpy.data.meshes.new(f"{obj.name}_{role}_Proxy")
    proxy = None
    try:
        mesh.vertices.add(len(co))
        mesh.vertices.foreach_set("co", co.astype(np.float32).ravel())
        mesh.loops.add(len(tri) * 3)
        mesh.loops.foreach_set("vertex_index", tri.astype(np.int32).ravel())
        mesh.polygons.add(len(tri))
        mesh.polygons.foreach_set("loop_start", np.arange(len(tri), dtype=np.int32) * 3)
        mesh.polygons.foreach_set("loop_total", np.full(len(tri), 3, np.int32))
        mesh.polygons.foreach_set("use_smooth", np.ones(len(tri), dtype=np.bool_))
        mesh.update(calc_edges=True)
        proxy = bpy.data.objects.new(mesh.name, mesh)
        proxy[_OWNER] = owner
        context.collection.objects.link(proxy)
        if len(tri) > face_budget:
            # Collapse connected edges, never weld separate clothing layers by proximity.
            reducer = proxy.modifiers.new("Proxy Reduction", "DECIMATE")
            reducer.ratio = face_budget / len(tri)
            reducer.use_collapse_triangulate = True
            depsgraph = context.evaluated_depsgraph_get()
            reduced = bpy.data.meshes.new_from_object(proxy.evaluated_get(depsgraph), depsgraph=depsgraph)
            proxy.modifiers.remove(reducer)
            proxy.data = reduced
            bpy.data.meshes.remove(mesh)
            mesh = reduced
            mesh.name = proxy.name
            if not len(mesh.polygons):
                raise ValueError(f"{obj.name}: proxy resolution is too low for this surface.")
        proxy.hide_render = True
        proxy.display_type = "WIRE"
        proxy.show_in_front = True
        proxy.color = (.1, .6, 1, 1) if role == "Source" else (1, .3, .1, 1)
        space = context.space_data
        if space and space.type == "VIEW_3D" and space.local_view:
            proxy.local_view_set(space, True)
        return proxy
    except Exception:
        if proxy:
            bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise


def sync_shrinkwrap(settings):
    """Synchronize native modifier settings without scanning geometry or weights."""
    obj = settings.source_proxy
    modifier = obj.modifiers.get(settings.wrap_name) if obj else None
    if modifier and modifier.type == "SHRINKWRAP":
        modifier.target = settings.target_proxy
        modifier.wrap_method = settings.method
        modifier.wrap_mode = settings.snap_mode
        modifier.offset = settings.offset
        modifier.vertex_group = settings.proxy_group
        modifier.invert_vertex_group = settings.invert_group
        modifier.use_project_x = settings.project_x
        modifier.use_project_y = settings.project_y
        modifier.use_project_z = settings.project_z
        modifier.use_positive_direction = settings.positive
        modifier.use_negative_direction = settings.negative
        modifier.project_limit = settings.project_limit
    source = settings.source
    lattice = source.modifiers.get(settings.modifier_name) if source else None
    if lattice and lattice.type == "LATTICE" and lattice.object == settings.lattice:
        lattice.vertex_group = settings.source_group
        lattice.invert_vertex_group = settings.invert_source_group


def _validate_masks(settings):
    for obj, name, label in ((settings.source, settings.source_group, "Source Clothing"),
                             (settings.source_proxy, settings.proxy_group, "Source Proxy")):
        if name and (obj is None or obj.vertex_groups.get(name) is None):
            raise ValueError(f"Vertex group '{name}' does not exist on {label}. Choose an existing mask or clear it.")


def create_proxies(context):
    _require_numpy("Proxy Shrinkwrap")
    s = context.scene.mdt_proxy_fit_settings
    if s.owner:
        raise ValueError("Apply or cancel the existing proxy fit first.")
    source, target = s.source, s.target
    if source is None or target is None:
        selected = [o for o in context.selected_objects if o.type == "MESH"]
        if len(selected) == 2 and context.object in selected:
            target = context.object
            source = next(o for o in selected if o != target)
    if not source or not target or source == target or source.type != "MESH" or target.type != "MESH":
        raise ValueError("Choose different source and target meshes, or select two meshes with the target active.")
    if not source.is_editable:
        raise ValueError("The source object must be editable.")
    if abs(source.matrix_world.to_3x3().determinant()) < 1e-20:
        raise ValueError("The source object scale must be nonzero on every axis.")
    s.source, s.target = source, target
    s.owner = uuid.uuid4().hex
    s.previous_active = context.object
    s.selection = json.dumps([o.name for o in context.selected_objects])
    try:
        s.source_proxy = _make_proxy(context, source, s.proxy_faces, s.owner, "Source")
        s.target_proxy = _make_proxy(context, target, s.proxy_faces, s.owner, "Target")
        modifier = s.source_proxy.modifiers.new("Proxy Shrinkwrap", "SHRINKWRAP")
        s.wrap_name = modifier.name
        modifier.show_viewport = modifier.show_render = False
        sync_shrinkwrap(s)
        s.report = "Edit either proxy and paint groups on Source Proxy, then Bind & Preview."
    except Exception:
        cancel_fit(context)
        raise
    return len(s.source_proxy.data.polygons), len(s.target_proxy.data.polygons)


def _stencil(points, low, step, shape):
    """Eight trilinear basis weights per point; no dense points-by-cage matrix."""
    uvw = np.clip((points - low) / step, 0, np.asarray(shape) - 1)
    cell = np.minimum(np.floor(uvw).astype(np.int32), np.asarray(shape, np.int32) - 2)
    fraction = uvw - cell
    indices, basis = [], []
    for dx, dy, dz in itertools.product(range(2), repeat=3):
        i = cell + (dx, dy, dz)
        indices.append(i[:, 0] + shape[0] * (i[:, 1] + shape[1] * i[:, 2]))
        basis.append(np.prod(np.where(np.array((dx, dy, dz)), fraction, 1 - fraction), axis=1))
    return np.column_stack(indices), np.column_stack(basis)


def _grid_edges(shape):
    grid = np.arange(np.prod(shape)).reshape(tuple(reversed(shape)))
    chunks = []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        chunks.append(np.column_stack((grid[tuple(a)].ravel(), grid[tuple(b)].ravel())))
    return np.concatenate(chunks)


def _fit_field(points, delta, low, step, shape, stiffness):
    ids, basis = _stencil(points, low, step, shape)
    count = int(np.prod(shape))
    edges = _grid_edges(shape)
    degree = np.bincount(edges.ravel(), minlength=count)
    regularizer = stiffness * max(len(points) / count, .1)
    ridge = 1e-6

    def transpose(values):
        out = np.zeros((count, 3), np.float64)
        np.add.at(out, ids.ravel(), (values[:, None, :] * basis[:, :, None]).reshape(-1, 3))
        return out

    def multiply(values):
        sampled = np.einsum("pi,pij->pj", basis, values[ids])
        out = transpose(sampled) + values * (regularizer * degree[:, None] + ridge)
        np.add.at(out, edges[:, 0], -regularizer * values[edges[:, 1]])
        np.add.at(out, edges[:, 1], -regularizer * values[edges[:, 0]])
        return out

    diagonal = np.bincount(ids.ravel(), weights=(basis * basis).ravel(), minlength=count)
    diagonal += regularizer * degree + ridge
    rhs = transpose(delta)
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    z = residual / diagonal[:, None]
    direction = z.copy()
    rz = float(np.sum(residual * z))
    tolerance = max(float(np.linalg.norm(rhs)) * 1e-6, 1e-10)
    iterations = 0
    for iterations in range(120):
        if np.linalg.norm(residual) <= tolerance:
            break
        product = multiply(direction)
        alpha = rz / max(float(np.sum(direction * product)), 1e-30)
        x += alpha * direction
        residual -= alpha * product
        z = residual / diagonal[:, None]
        next_rz = float(np.sum(residual * z))
        direction = z + (next_rz / max(rz, 1e-30)) * direction
        rz = next_rz
    fitted = np.einsum("pi,pij->pj", basis, x[ids])
    return x, float(np.sqrt(np.mean(np.sum((fitted - delta) ** 2, axis=1)))), iterations + 1


def _safe_strength(field, shape, step, requested):
    grid = field.reshape(tuple(reversed(shape)) + (3,))
    cell = tuple(n - 1 for n in reversed(shape))
    gradients = []
    for x, y, z in itertools.product(range(2), repeat=3):
        du = (grid[z:z + cell[0], y:y + cell[1], 1:, :] - grid[z:z + cell[0], y:y + cell[1], :-1, :]) / step[0]
        dv = (grid[z:z + cell[0], 1:, x:x + cell[2], :] - grid[z:z + cell[0], :-1, x:x + cell[2], :]) / step[1]
        dw = (grid[1:, y:y + cell[1], x:x + cell[2], :] - grid[:-1, y:y + cell[1], x:x + cell[2], :]) / step[2]
        gradients.append(np.stack((du, dv, dw), axis=-1).reshape(-1, 3, 3))
    gradients = np.concatenate(gradients)
    strength = requested
    for _ in range(24):
        jacobian = np.eye(3) + gradients * strength
        if np.min(np.linalg.det(jacobian)) > .025 and np.min(np.linalg.svd(jacobian, compute_uv=False)[:, -1]) > .15:
            return strength
        strength *= .8
    raise ValueError("The requested proxy deformation folds the lattice. Reduce the proxy movement or increase smoothness.")


def _constrain_movement(vectors, axes, space, matrix):
    """Constrain proxy samples and cage displacements in the same coordinate space."""
    if all(axes):
        return vectors
    mask = np.asarray(axes, dtype=np.float64)
    if space == "WORLD":
        return vectors * mask
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    if abs(np.linalg.det(linear)) < 1e-20:
        raise ValueError("The source object scale must be nonzero on every axis.")
    return ((vectors @ np.linalg.inv(linear).T) * mask) @ linear.T


def bind_fit(context):
    _require_numpy("Proxy Shrinkwrap")
    s = context.scene.mdt_proxy_fit_settings
    if s.lattice:
        raise ValueError("This proxy is already bound. Use Update Preview or Unbind.")
    if not s.source or not s.source_proxy or not s.target_proxy:
        raise ValueError("Create both proxies first.")
    _validate_masks(s)
    sync_shrinkwrap(s)
    proxy = s.source_proxy
    wrap = proxy.modifiers.get(s.wrap_name)
    if not wrap or wrap.type != "SHRINKWRAP":
        raise ValueError("The source proxy Shrinkwrap modifier is missing. Cancel and create proxies again.")
    wrap.show_viewport = False
    context.view_layer.update()
    start = _coordinates(proxy)
    mesh = proxy.data
    for name in (_REST, _IDS):
        attr = mesh.attributes.get(name)
        if attr:
            mesh.attributes.remove(attr)
    mesh.attributes.new(_REST, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", start.astype(np.float32).ravel())
    mesh.attributes.new(_IDS, "INT", "POINT").data.foreach_set("value", np.arange(len(start), dtype=np.int32))
    s.topology_hash = mesh_topology_signature(mesh, include_edges=True)
    source_co = _world(s.source, _mirror_coordinate_array(s.source))
    lower = np.minimum(source_co.min(0), start.min(0))
    upper = np.maximum(source_co.max(0), start.max(0))
    extent = upper - lower
    largest = max(float(extent.max()), 1e-6)
    extent = np.maximum(extent, largest * .08) * 1.12
    center = (upper + lower) * .5
    shape = np.maximum(3, np.rint(s.grid_detail * extent / extent.max()).astype(int))
    lattice = bpy.data.lattices.new("Proxy Fit Lattice")
    cage = None
    try:
        lattice.points_u, lattice.points_v, lattice.points_w = map(int, shape)
        lattice.interpolation_type_u = lattice.interpolation_type_v = lattice.interpolation_type_w = "KEY_LINEAR"
        cage = bpy.data.objects.new("Proxy Fit Lattice", lattice)
        cage[_OWNER] = s.owner
        context.collection.objects.link(cage)
        cage.hide_render = True
        cage.hide_set(True)
        cage.matrix_world = Matrix.Translation(center) @ Matrix.Diagonal(tuple(extent / (shape - 1)) + (1.,))
        s.lattice = cage
        s.grid_min = center - extent * .5
        s.grid_step = extent / (shape - 1)
        modifier = s.source.modifiers.new("Proxy Fit Lattice", "LATTICE")
        modifier.object = cage
        s.modifier_name = modifier.name
        run_object_operator(context, s.source, "modifier_move_to_index", modifier=modifier.name, index=0)
        wrap.show_viewport = True
        update_fit(context)
    except Exception:
        unbind_fit(context)
        if cage is None and lattice.users == 0:
            bpy.data.lattices.remove(lattice)
        raise


def update_fit(context):
    _require_numpy("Proxy Shrinkwrap")
    s = context.scene.mdt_proxy_fit_settings
    if not s.source_proxy or not s.target_proxy or not s.lattice:
        raise ValueError("Bind the proxy before updating.")
    if not s.source:
        raise ValueError("The original source mesh is missing. Cancel this proxy fit.")
    modifier = s.source.modifiers.get(s.modifier_name)
    if not modifier or modifier.type != "LATTICE" or modifier.object != s.lattice:
        raise ValueError("The source lattice modifier is missing. Unbind and Bind again.")
    proxy = s.source_proxy
    if mesh_topology_signature(proxy.data, include_edges=True) != s.topology_hash:
        raise ValueError("Source proxy topology changed after binding. Unbind, edit, then Bind again.")
    ids = proxy.data.attributes.get(_IDS)
    rest = proxy.data.attributes.get(_REST)
    if ids is None or rest is None:
        raise ValueError("Proxy correspondence is missing. Unbind and Bind again.")
    values = np.empty(len(proxy.data.vertices), np.int32)
    ids.data.foreach_get("value", values)
    if not np.array_equal(values, np.arange(len(values))):
        raise ValueError("Proxy correspondence changed. Unbind and Bind again.")
    _validate_masks(s)
    if s.method == "PROJECT" and (not (s.project_x or s.project_y or s.project_z) or not (s.positive or s.negative)):
        raise ValueError("Projection needs an axis and at least one direction.")
    sync_shrinkwrap(s)
    context.view_layer.update()
    points = np.empty((len(values), 3), np.float32)
    rest.data.foreach_get("vector", points.ravel())
    deformed = _coordinates(proxy, evaluated=True)
    if deformed.shape != points.shape or not np.isfinite(deformed).all():
        raise ValueError("The evaluated source proxy must retain its vertex correspondence and finite coordinates.")
    lattice = s.lattice.data
    shape = (lattice.points_u, lattice.points_v, lattice.points_w)
    delta = _constrain_movement(deformed - points, s.movement_axes, s.movement_space, s.source.matrix_world)
    field, rms, iterations = _fit_field(points, delta, np.array(s.grid_min), np.array(s.grid_step), shape, s.smoothness)
    # Enforce the constraint on the final cage as well as the input samples;
    # regularization and numerical tolerances must not reintroduce locked motion.
    field = _constrain_movement(field, s.movement_axes, s.movement_space, s.source.matrix_world)
    strength = _safe_strength(field, shape, np.array(s.grid_step), s.strength)
    base = np.empty((len(lattice.points), 3), np.float32)
    lattice.points.foreach_get("co", base.ravel())
    # The native lattice has unit-spaced local points; grid_step converts meters.
    lattice.points.foreach_set("co_deform", (base + field * strength / np.array(s.grid_step)).astype(np.float32).ravel())
    context.view_layer.update()
    s.report = f"Strength {strength:.3f}/{s.strength:.3f}; proxy fit RMS {rms:.6g}; {iterations} solver steps."
    return {"strength": strength, "rms": rms, "iterations": iterations}


def _remove_owned(obj, owner):
    if not obj or obj.get(_OWNER) != owner:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data.users == 0:
        (bpy.data.lattices if isinstance(data, bpy.types.Lattice) else bpy.data.meshes).remove(data)


def unbind_fit(context):
    s = context.scene.mdt_proxy_fit_settings
    modifier = s.source.modifiers.get(s.modifier_name) if s.source else None
    if modifier and modifier.type == "LATTICE" and modifier.object == s.lattice:
        s.source.modifiers.remove(modifier)
    cage = s.lattice
    s.lattice = None
    _remove_owned(cage, s.owner)
    wrap = s.source_proxy.modifiers.get(s.wrap_name) if s.source_proxy else None
    if wrap:
        wrap.show_viewport = False
    s.modifier_name = s.topology_hash = ""
    s.report = "Edit proxies or lattice resolution, then Bind & Preview."


def cancel_fit(context):
    s = context.scene.mdt_proxy_fit_settings
    owner, selected, active = s.owner, json.loads(s.selection or "[]"), s.previous_active
    unbind_fit(context)
    source_proxy, target_proxy = s.source_proxy, s.target_proxy
    s.source_proxy = s.target_proxy = None
    _remove_owned(source_proxy, owner)
    _remove_owned(target_proxy, owner)
    s.owner = s.wrap_name = s.selection = s.report = ""
    s.previous_active = None
    for obj in list(context.selected_objects):
        obj.select_set(False)
    for name in selected:
        obj = context.view_layer.objects.get(name)
        if obj and obj.visible_get():
            obj.select_set(True)
    if active and context.view_layer.objects.get(active.name):
        context.view_layer.objects.active = active


def apply_fit(context):
    s = context.scene.mdt_proxy_fit_settings
    if not s.source or not s.lattice:
        raise ValueError("Bind and preview the proxy fit first.")
    modifier = s.source.modifiers.get(s.modifier_name)
    if not modifier or modifier.type != "LATTICE" or modifier.object != s.lattice or s.source.modifiers.find(modifier.name) != 0:
        raise ValueError("Keep the Proxy Fit lattice first in the source modifier stack before applying.")
    result = update_fit(context)
    source = s.source
    # Freeze the current per-vertex lattice movement into Basis and every key.
    # This keeps both the current mix and original relative key deltas intact;
    # separately deforming each key would make nonlinear mixed values jump.
    _points, movement = fitted_source_displacements(s)
    delta = (movement @ np.linalg.inv(np.asarray(source.matrix_world)[:3, :3]).T).astype(np.float32)
    # A shared datablock must not change unselected instances.
    original_mesh = source.data
    if not original_mesh.is_editable:
        raise ValueError("Make the source mesh data local before applying.")
    if original_mesh.users > 1:
        source.data = original_mesh.copy()
    mesh = source.data
    targets = mesh_coordinate_offset_plan(mesh, delta)
    try:
        for vertices, _before, after in targets:
            vertices.foreach_set("co", after.ravel())
        mesh.update()
    except Exception:
        for vertices, before, _after in targets:
            vertices.foreach_set("co", before.ravel())
        mesh.update()
        if mesh != original_mesh:
            source.data = original_mesh
            bpy.data.meshes.remove(mesh)
        raise
    source.modifiers.remove(modifier)
    cancel_fit(context)
    return result


def fitted_source_displacements(settings):
    """Sample only our first lattice, excluding bone poses and later modifiers."""
    source, cage = settings.source, settings.lattice
    modifier = source.modifiers.get(settings.modifier_name) if source else None
    if (not cage or not modifier or modifier.type != "LATTICE" or modifier.object != cage
            or source.modifiers.find(modifier.name) != 0 or not modifier.show_viewport):
        raise ValueError("Keep the enabled Proxy Fit lattice first in the source modifier stack.")
    points = _world(source, _mirror_coordinate_array(source))
    lattice = cage.data
    shape = (lattice.points_u, lattice.points_v, lattice.points_w)
    base = np.empty((len(lattice.points), 3), np.float32)
    moved = np.empty_like(base)
    lattice.points.foreach_get("co", base.ravel())
    lattice.points.foreach_get("co_deform", moved.ravel())
    field = (moved - base) * np.array(settings.grid_step)
    ids, basis = _stencil(points, np.array(settings.grid_min), np.array(settings.grid_step), shape)
    movement = np.einsum("pi,pij->pj", basis, field[ids])
    if modifier.vertex_group:
        weights = _flat_vertex_group_weights(source, modifier.vertex_group, len(points), {})
        if modifier.invert_vertex_group:
            weights = 1 - weights
        movement *= weights[:, None]
    return points, movement
