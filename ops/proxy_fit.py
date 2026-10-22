"""Explicit session actions keep expensive proxy fitting out of UI callbacks."""

import bpy

from ..core.common import ensure_object_mode, restore_context_mode
from ..core import proxy_fit


class MDT_OT_proxy_fit(bpy.types.Operator):
    bl_idname = "mdt.proxy_fit"
    bl_label = "Proxy Shrinkwrap"
    bl_description = "Fit editable proxies and transfer their movement through a volume lattice"
    bl_options = {"REGISTER", "UNDO"}

    action: bpy.props.EnumProperty(items=(
        ("CREATE", "Create Proxies", "Create editable modeling-space source and target proxies"),
        ("BIND", "Bind & Preview", "Record the source proxy rest state and fit a volume lattice"),
        ("UPDATE", "Update Preview", "Update clothing after editing a proxy, mask or Shrinkwrap settings"),
        ("UNBIND", "Unbind", "Remove the preview lattice and keep proxies for topology editing"),
        ("APPLY", "Apply", "Bake current movement into Basis and every key, retaining shape-key deltas and later modifiers"),
        ("CANCEL", "Cancel", "Remove this fit's helpers and preview modifier"),
        ("SOURCE", "Select Source Proxy", "Select the editable clothing control surface"),
        ("TARGET", "Select Target Proxy", "Select the editable body control surface"),
    ))

    def execute(self, context):
        previous = context.mode
        try:
            ensure_object_mode(context)
            if self.action in {"SOURCE", "TARGET"}:
                settings = context.scene.mdt_proxy_fit_settings
                obj = settings.source_proxy if self.action == "SOURCE" else settings.target_proxy
                if obj is None or context.view_layer.objects.get(obj.name) is None:
                    raise ValueError("The proxy is missing from this view layer.")
                obj.hide_set(False)
                for selected in list(context.selected_objects):
                    selected.select_set(False)
                obj.select_set(True)
                context.view_layer.objects.active = obj
            else:
                operations = {
                    "CREATE": proxy_fit.create_proxies, "BIND": proxy_fit.bind_fit,
                    "UPDATE": proxy_fit.update_fit, "UNBIND": proxy_fit.unbind_fit,
                    "APPLY": proxy_fit.apply_fit, "CANCEL": proxy_fit.cancel_fit,
                }
                operations[self.action](context)
                # An update while editing/painting a proxy keeps that interaction mode.
                if self.action in {"UPDATE", "BIND", "UNBIND"}:
                    restore_context_mode(previous)
        except (ValueError, RuntimeError) as exc:
            restore_context_mode(previous)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if self.action in {"BIND", "UPDATE"}:
            self.report({"INFO"}, context.scene.mdt_proxy_fit_settings.report)
        return {"FINISHED"}
