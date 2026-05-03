"""UI lists for Mod Tool."""

import bpy


class MDT_UL_group_match_results(bpy.types.UIList):
    """Compact list of source-to-target vertex-group rename suggestions."""

    _STATUS_ICONS = {
        "DIRECT": "CHECKMARK",
        "MANUAL": "GREASEPENCIL",
        "AMBIGUOUS": "QUESTION",
        "UNMATCHED": "ERROR",
        "CONFLICT": "CANCEL",
    }

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index=0, _flt_flag=0):
        status_icon = self._STATUS_ICONS.get(item.status, "INFO")

        row = layout.row(align=True)
        row.label(text=item.source_group or "-", icon="GROUP_VERTEX")
        row.label(text="->")
        row.label(text=item.target_group or "-", icon="GROUP_VERTEX" if item.target_group else status_icon)
        row.label(text=item.status.title(), icon=status_icon)
        row.label(text=f"{item.confidence:.2f}")
