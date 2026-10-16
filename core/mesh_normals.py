"""Mesh custom-normal helpers for selected seam-style vertices."""

from __future__ import annotations

from array import array
from collections import defaultdict
from math import floor

from mathutils import Vector

from .common import is_mesh_object, sync_edit_mesh_data


DEFAULT_NORMAL_MERGE_DISTANCE = 1.0e-4
NORMAL_MERGE_AVERAGE = "AVERAGE"
NORMAL_MERGE_ACTIVE = "ACTIVE"
_NORMAL_EPSILON = 1.0e-8


class _DisjointSet:
    def __init__(self, size):
        self.parents = list(range(size))
        self.ranks = [0] * size

    def find(self, item):
        parent = self.parents[item]
        if parent != item:
            parent = self.find(parent)
            self.parents[item] = parent
        return parent

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return

        rank_a = self.ranks[root_a]
        rank_b = self.ranks[root_b]
        if rank_a < rank_b:
            self.parents[root_a] = root_b
        elif rank_a > rank_b:
            self.parents[root_b] = root_a
        else:
            self.parents[root_b] = root_a
            self.ranks[root_a] += 1


def _cell_key(world_coordinate, distance):
    inverse_distance = 1.0 / distance
    return (
        floor(world_coordinate.x * inverse_distance),
        floor(world_coordinate.y * inverse_distance),
        floor(world_coordinate.z * inverse_distance),
    )


def _neighbor_keys(base_key):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield (base_x + offset_x, base_y + offset_y, base_z + offset_z)


def _read_selected_vertex_indices(mesh):
    selected_flags = array("b", [0]) * len(mesh.vertices)
    if selected_flags:
        mesh.vertices.foreach_get("select", selected_flags)
    return [vertex_index for vertex_index, is_selected in enumerate(selected_flags) if is_selected]


def _read_vertex_coordinates(mesh):
    coordinates = array("f", [0.0]) * (len(mesh.vertices) * 3)
    if coordinates:
        mesh.vertices.foreach_get("co", coordinates)
    return coordinates


def _read_loop_vertex_indices(mesh):
    loop_vertex_indices = array("i", [0]) * len(mesh.loops)
    if loop_vertex_indices:
        mesh.loops.foreach_get("vertex_index", loop_vertex_indices)
    return loop_vertex_indices


def _read_corner_normals(mesh):
    normals = array("f", [0.0]) * (len(mesh.loops) * 3)
    if normals:
        mesh.corner_normals.foreach_get("vector", normals)
    return normals


def _local_normal_from_flat(normals, loop_index):
    offset = loop_index * 3
    normal = Vector((normals[offset], normals[offset + 1], normals[offset + 2]))
    if normal.length <= _NORMAL_EPSILON:
        return None
    normal.normalize()
    return normal


def _world_normal_from_local(local_normal, normal_matrix):
    world_normal = normal_matrix @ local_normal
    if world_normal.length <= _NORMAL_EPSILON:
        return None
    world_normal.normalize()
    return world_normal


def _local_normal_from_world(world_normal, local_normal_matrix):
    local_normal = local_normal_matrix @ world_normal
    if local_normal.length <= _NORMAL_EPSILON:
        return None
    local_normal.normalize()
    return local_normal


def _build_loop_indices_by_vertex(loop_vertex_indices):
    loop_indices_by_vertex = defaultdict(list)
    for loop_index, vertex_index in enumerate(loop_vertex_indices):
        loop_indices_by_vertex[vertex_index].append(loop_index)
    return loop_indices_by_vertex


def _collect_object_normal_points(obj, object_index):
    if not is_mesh_object(obj):
        return [], None

    sync_edit_mesh_data(obj)
    mesh = obj.data
    if not mesh.vertices or not mesh.loops:
        return [], None

    vertex_indices = _read_selected_vertex_indices(mesh)
    if not vertex_indices:
        return [], None

    mesh.update()
    vertex_coordinates = _read_vertex_coordinates(mesh)
    loop_vertex_indices = _read_loop_vertex_indices(mesh)
    loop_normals = _read_corner_normals(mesh)
    loop_indices_by_vertex = _build_loop_indices_by_vertex(loop_vertex_indices)
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    object_data = {
        "object": obj,
        "loop_normals": loop_normals,
        "loop_indices_by_vertex": loop_indices_by_vertex,
        "local_normal_matrix": obj.matrix_world.to_3x3().transposed(),
    }
    points = []

    for vertex_index in vertex_indices:
        coordinate_offset = vertex_index * 3
        world_coordinate = obj.matrix_world @ Vector(
            (
                vertex_coordinates[coordinate_offset],
                vertex_coordinates[coordinate_offset + 1],
                vertex_coordinates[coordinate_offset + 2],
            )
        )

        world_normal_sum = Vector((0.0, 0.0, 0.0))
        loop_indices = loop_indices_by_vertex.get(vertex_index, ())
        for loop_index in loop_indices:
            local_normal = _local_normal_from_flat(loop_normals, loop_index)
            if local_normal is None:
                continue
            world_normal = _world_normal_from_local(local_normal, normal_matrix)
            if world_normal is not None:
                world_normal_sum += world_normal

        if world_normal_sum.length <= _NORMAL_EPSILON:
            continue
        world_normal_sum.normalize()
        points.append(
            {
                "object_index": object_index,
                "vertex_index": vertex_index,
                "world_coordinate": world_coordinate,
                "world_normal": world_normal_sum,
                "loop_indices": tuple(loop_indices),
            }
        )

    return points, object_data


def _collect_selected_normal_points(mesh_objects):
    points = []
    object_data = {}
    active_object_index = None

    for object_index, obj in enumerate(mesh_objects):
        sync_edit_mesh_data(obj)
        object_points, data = _collect_object_normal_points(obj, object_index)
        if data is not None:
            object_data[object_index] = data
        points.extend(object_points)

    return points, object_data


def _build_distance_clusters(points, merge_distance):
    disjoint_set = _DisjointSet(len(points))
    spatial_hash = {}
    merge_distance_squared = merge_distance * merge_distance

    for point_index, point in enumerate(points):
        world_coordinate = point["world_coordinate"]
        base_key = _cell_key(world_coordinate, merge_distance)
        for neighbor_key in _neighbor_keys(base_key):
            for other_index in spatial_hash.get(neighbor_key, ()):
                other_coordinate = points[other_index]["world_coordinate"]
                if (world_coordinate - other_coordinate).length_squared <= merge_distance_squared:
                    disjoint_set.union(point_index, other_index)
        spatial_hash.setdefault(base_key, []).append(point_index)

    clusters = defaultdict(list)
    for point_index in range(len(points)):
        clusters[disjoint_set.find(point_index)].append(point_index)
    return [cluster for cluster in clusters.values() if len(cluster) > 1]


def _average_cluster_normal(points, cluster):
    normal_sum = Vector((0.0, 0.0, 0.0))
    for point_index in cluster:
        normal_sum += points[point_index]["world_normal"]
    if normal_sum.length <= _NORMAL_EPSILON:
        fallback_normal = points[cluster[0]]["world_normal"].copy()
        return fallback_normal if fallback_normal.length > _NORMAL_EPSILON else None
    normal_sum.normalize()
    return normal_sum


def _active_cluster_normal(points, cluster, active_object_index):
    if active_object_index is None:
        return None
    active_point_indices = [
        point_index
        for point_index in cluster
        if points[point_index]["object_index"] == active_object_index
    ]
    if not active_point_indices:
        return None
    return _average_cluster_normal(points, active_point_indices)


def _cluster_normal(points, cluster, mode, active_object_index):
    if mode == NORMAL_MERGE_AVERAGE:
        return _average_cluster_normal(points, cluster)
    if mode == NORMAL_MERGE_ACTIVE:
        return _active_cluster_normal(points, cluster, active_object_index)
    raise ValueError(f"Unsupported normal merge mode: {mode}")


def _write_merged_normals(points, clusters, object_data, mode, active_object_index):
    touched_vertices = set()
    touched_objects = set()
    written_loop_count = 0
    skipped_cluster_count = 0

    for cluster in clusters:
        world_normal = _cluster_normal(points, cluster, mode, active_object_index)
        if world_normal is None:
            skipped_cluster_count += 1
            continue

        for point_index in cluster:
            point = points[point_index]
            data = object_data[point["object_index"]]
            local_normal = _local_normal_from_world(world_normal, data["local_normal_matrix"])
            if local_normal is None:
                continue

            for loop_index in point["loop_indices"]:
                offset = loop_index * 3
                data["loop_normals"][offset] = local_normal.x
                data["loop_normals"][offset + 1] = local_normal.y
                data["loop_normals"][offset + 2] = local_normal.z
                written_loop_count += 1

            touched_vertices.add((point["object_index"], point["vertex_index"]))
            touched_objects.add(point["object_index"])

    for object_index in touched_objects:
        data = object_data[object_index]
        mesh = data["object"].data
        flat_normals = data["loop_normals"]
        loop_normals = [
            (
                flat_normals[offset],
                flat_normals[offset + 1],
                flat_normals[offset + 2],
            )
            for offset in range(0, len(flat_normals), 3)
        ]
        mesh.normals_split_custom_set(loop_normals)
        mesh.update()

    return {
        "touched_object_count": len(touched_objects),
        "touched_vertex_count": len(touched_vertices),
        "written_loop_count": written_loop_count,
        "skipped_cluster_count": skipped_cluster_count,
    }


def merge_selected_vertex_normals(
    mesh_objects,
    merge_distance=DEFAULT_NORMAL_MERGE_DISTANCE,
    mode=NORMAL_MERGE_AVERAGE,
    active_obj=None,
):
    """Unify custom normals for selected vertices grouped by world-space distance."""
    if merge_distance <= 0.0:
        raise ValueError("Merge distance must be greater than zero.")
    if mode not in {NORMAL_MERGE_AVERAGE, NORMAL_MERGE_ACTIVE}:
        raise ValueError(f"Unsupported normal merge mode: {mode}")

    mesh_objects = [obj for obj in mesh_objects if is_mesh_object(obj)]
    if not mesh_objects:
        raise ValueError("Select at least one mesh object.")
    active_object_index = None
    if mode == NORMAL_MERGE_ACTIVE:
        if not is_mesh_object(active_obj):
            raise ValueError("The active object must be a mesh object.")
        for object_index, obj in enumerate(mesh_objects):
            if obj == active_obj:
                active_object_index = object_index
                break
        if active_object_index is None:
            raise ValueError("The active object must be one of the selected mesh objects.")

    points, object_data = _collect_selected_normal_points(mesh_objects)
    if len(points) < 2:
        raise ValueError("Select at least two connected or nearby mesh vertices.")

    clusters = _build_distance_clusters(points, merge_distance)
    if not clusters:
        return {
            "selected_vertex_count": len(points),
            "merged_cluster_count": 0,
            "touched_object_count": 0,
            "touched_vertex_count": 0,
            "written_loop_count": 0,
            "skipped_cluster_count": 0,
        }

    result = _write_merged_normals(points, clusters, object_data, mode, active_object_index)
    result.update(
        {
            "selected_vertex_count": len(points),
            "merged_cluster_count": len(clusters),
        }
    )
    return result
