"""Material operators for Mod Tool."""

import bpy

from ..core.common import ensure_object_mode, get_selected_mesh_objects, is_mesh_object, restore_context_mode
from ..core.materials import assign_single_material_to_object, rename_object_from_material


class MDT_OT_create_named_materials(bpy.types.Operator):
    """Create one material per selected mesh and name it after the object."""

    bl_idname = "mdt.create_named_materials"
    bl_label = "Create Object Materials"
    bl_description = "Replace each selected mesh object's materials with one material named after the object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        mesh_objects = get_selected_mesh_objects(context)
        if not mesh_objects:
            self.report({"ERROR"}, "Select at least one mesh object.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        processed_count = 0
        created_count = 0

        try:
            for obj in mesh_objects:
                material = bpy.data.materials.get(obj.name)
                if material is None:
                    material = bpy.data.materials.new(name=obj.name)
                    created_count += 1

                assign_single_material_to_object(obj, material)
                processed_count += 1
        finally:
            restore_context_mode(previous_mode)

        self.report(
            {"INFO"},
            f"Assigned one material to {processed_count} object(s). Created {created_count} new material(s).",
        )
        return {"FINISHED"}


class MDT_OT_separate_by_material(bpy.types.Operator):
    """Separate each selected mesh object by material and rename pieces after the material."""

    bl_idname = "mdt.separate_by_material"
    bl_label = "Split By Material"
    bl_description = "Separate each selected mesh object by material and rename each result after the material"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, _context):
        return True

    def execute(self, context):
        mesh_objects = get_selected_mesh_objects(context)
        if not mesh_objects:
            self.report({"ERROR"}, "Select at least one mesh object.")
            return {"CANCELLED"}

        try:
            previous_mode = ensure_object_mode(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not switch to Object Mode: {exc}")
            return {"CANCELLED"}

        view_layer = context.view_layer
        separated_objects = []
        skipped_count = 0
        renamed_count = 0

        try:
            for obj in mesh_objects:
                if obj.data is None or len(obj.data.polygons) == 0:
                    skipped_count += 1
                    continue

                has_material = any(slot.material is not None for slot in obj.material_slots)
                if not has_material:
                    skipped_count += 1
                    continue

                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                view_layer.objects.active = obj

                try:
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.separate(type="MATERIAL")
                    bpy.ops.object.mode_set(mode="OBJECT")
                except RuntimeError as exc:
                    bpy.ops.object.mode_set(mode="OBJECT")
                    self.report({"ERROR"}, f"Could not split {obj.name} by material: {exc}")
                    return {"CANCELLED"}

                pieces = [piece for piece in context.selected_objects if is_mesh_object(piece)]
                if not pieces:
                    pieces = [obj]

                for piece in pieces:
                    if rename_object_from_material(piece):
                        renamed_count += 1
                    separated_objects.append(piece)
        finally:
            restore_context_mode(previous_mode)

        if separated_objects:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in separated_objects:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            if separated_objects[0].name in bpy.data.objects:
                view_layer.objects.active = separated_objects[0]

        self.report(
            {"INFO"},
            f"Split selected meshes by material. Renamed {renamed_count} object(s). Skipped {skipped_count} object(s).",
        )
        return {"FINISHED"}
