"""Shared sparse weight snapshots and rollback-capable writes."""

import hashlib
import json

from .materials import np
from .rename import _write_group_weight_array


def snapshot_weights(obj):
    """Read sparse RNA memberships once, retaining names, locks and group order."""
    groups = [(g.name, g.lock_weight) for g in obj.vertex_groups]
    names = [name for name, _locked in groups]
    rows, entries = [], []
    for vertex in obj.data.vertices:
        row = {}
        for member in vertex.groups:
            if member.weight != 0:
                row[names[member.group]] = member.weight
                entries.append((vertex.index, member.group, member.weight))
        rows.append(row)
    digest = hashlib.sha256(json.dumps(groups).encode())
    if entries:
        values = np.asarray(entries, dtype=np.float64)
        if not np.isfinite(values[:, 2]).all() or np.any((values[:, 2] < 0) | (values[:, 2] > 1)):
            raise ValueError(f"{obj.name}: invalid vertex weights.")
        order = np.lexsort((values[:, 1], values[:, 0]))
        digest.update(values[order].tobytes())
    return {"groups": groups, "rows": rows, "hash": digest.hexdigest(),
            "active": obj.vertex_groups.active.name if obj.vertex_groups.active else None}


def write_weight_rows(obj, rows, changed, schema):
    """Write sparse rows in name batches; locks are retained as brush settings."""
    desired = dict(schema)
    for group in reversed(list(obj.vertex_groups)):
        if group.name not in desired:
            obj.vertex_groups.remove(group)
    groups = {g.name: g for g in obj.vertex_groups}
    for name, _locked in schema:
        if name not in groups:
            groups[name] = obj.vertex_groups.new(name=name)
    pending = {}
    for i in changed:
        for name, value in rows[i].items():
            pending.setdefault(name, []).append((i, value))
    for name, group in groups.items():
        if changed:
            group.remove(changed)
        entries = pending.get(name)
        if entries:
            values = np.asarray(entries, dtype=np.float64)
            _write_group_weight_array(group, values[:, 0].astype(np.int32), values[:, 1])
        group.lock_weight = desired[name]
    obj.data.update()


def restore_weight_snapshot(obj, snapshot):
    obj.vertex_groups.clear()
    write_weight_rows(obj, snapshot["rows"], list(range(len(snapshot["rows"]))), snapshot["groups"])
    if snapshot["active"] is not None:
        obj.vertex_groups.active_index = obj.vertex_groups.find(snapshot["active"])
