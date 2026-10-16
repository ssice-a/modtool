"""Shared Edit Mode edge-cycle selection helpers."""

import bmesh

from .common import is_mesh_object


def edit_mesh_objects(context):
    objects = [
        obj for obj in getattr(context, "objects_in_mode_unique_data", ())
        if is_mesh_object(obj) and obj.mode == "EDIT"
    ]
    if objects:
        return objects
    obj = getattr(context, "edit_object", None)
    return [obj] if is_mesh_object(obj) else []


def selected_edge_components(bm, *, boundary_only=False):
    selected = {
        edge for edge in bm.edges
        if edge.select and not edge.hide
        and (not boundary_only or len(edge.link_faces) == 1)
    }
    components = []
    while selected:
        seed = selected.pop()
        component = [seed]
        stack = [seed]
        while stack:
            edge = stack.pop()
            for vert in edge.verts:
                for linked in vert.link_edges:
                    if linked in selected:
                        selected.remove(linked)
                        stack.append(linked)
                        component.append(linked)
        components.append(component)
    return components


def active_edge_for_component(component, active_element):
    if isinstance(active_element, bmesh.types.BMEdge):
        return active_element if active_element in component else None
    if isinstance(active_element, bmesh.types.BMVert):
        return next((edge for edge in component if active_element in edge.verts), None)
    return None


def order_edge_cycle(component, start_edge=None):
    if len(component) < 3:
        raise ValueError("Each selected loop needs at least three edges.")
    adjacency = {}
    for edge in component:
        for vert in edge.verts:
            adjacency.setdefault(vert, []).append(edge)
    if len(adjacency) != len(component) or any(len(edges) != 2 for edges in adjacency.values()):
        raise ValueError("Each selection must be one closed, non-branching edge loop.")

    first_edge = start_edge if start_edge in component else component[0]
    first, second = first_edge.verts
    verts, edges = [first, second], [first_edge]
    visited = {first, second}
    previous, current = first_edge, second
    while len(edges) < len(component):
        edge = next(edge for edge in adjacency[current] if edge != previous)
        next_vert = edge.other_vert(current)
        edges.append(edge)
        if next_vert == first:
            break
        if next_vert in visited:
            raise ValueError("The selected loop crosses itself topologically.")
        verts.append(next_vert)
        visited.add(next_vert)
        previous, current = edge, next_vert
    if len(verts) != len(component) or edges[-1].other_vert(current) != first:
        raise ValueError("The selected loop is not closed.")
    return verts, edges
