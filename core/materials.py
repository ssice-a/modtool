"""Material helpers used by the modding workflow."""


def assign_single_material_to_object(obj, material):
    """Replace all material slots with one material and assign it to every polygon."""
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = 0


def resolve_used_material(obj):
    """Return the first material actually used by this mesh object's polygons."""
    used_indices = sorted({polygon.material_index for polygon in obj.data.polygons})
    for material_index in used_indices:
        if material_index >= len(obj.material_slots):
            continue
        material = obj.material_slots[material_index].material
        if material is not None:
            return material

    for slot in obj.material_slots:
        if slot.material is not None:
            return slot.material

    return None


def rename_object_from_material(obj):
    """Rename the object after its used material and normalize to a single slot."""
    material = resolve_used_material(obj)
    if material is None:
        return False

    assign_single_material_to_object(obj, material)
    obj.name = material.name
    obj.data.name = f"{obj.name}_Mesh"
    return True
