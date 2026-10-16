# SPDX-License-Identifier: AGPL-3.0-only
"""Bake standard MMD body controls onto a rest-pose FK skeleton.

MMD Tools owns file decoding. Retargeting is computed in arrays, without adding
control bones, constraints, mesh scans, or frame-change handlers to the scene.
Stance calibration and semi-standard folds adapted from Sasye/EIEM;
see THIRD_PARTY_NOTICES.md for source revision and licensing.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import unicodedata

import bpy
import numpy as np

from .common import is_armature_object
from .mmd_reference import MMD_TO_BLENDER, tda_reference


def _name(value):
    return unicodedata.normalize("NFKC", value).casefold().replace(" ", "")


def _target_name(value):
    return _name(value).replace("_", "").replace("-", "").replace(".", "")


SOURCE_NAMES = {
    "root": ("全ての親",), "center": ("センター",), "groove": ("グルーブ",),
    "waist": ("腰",), "pelvis": ("下半身",), "spine": ("上半身",),
    "chest_pre": ("上半身1",),
    "chest": ("上半身2", "上半身２"), "chest_extra": ("上半身3", "上半身３"),
    "neck": ("首",), "head": ("頭",),
}
TARGET_NAMES = {
    "root": ("Root", "root", "全ての親"),
    "pelvis": ("Bip001_Pelvis", "Bip001-Pelvis", "下半身", "Hips"),
    "spine": ("Bip001_Spine", "Bip001-Spine", "上半身", "Spine"),
    "chest": ("Bip001_Spine2", "Bip001-Spine2", "上半身2", "Chest"),
    "neck": ("Bip001_Neck", "Bip001-Neck", "首", "Neck"),
    "head": ("Bip001_Head", "Bip001-Head", "頭", "Head"),
}
PARENTS = {
    "root": None, "center": "root", "groove": "center",
    "pelvis": "groove", "spine": "groove", "chest": "spine",
    "neck": "chest", "head": "neck",
}
for _side, _jp in (("L", "左"), ("R", "右")):
    for _role, _mmd, _bip, _english, _parent in (
        ("clavicle", "肩", "Clavicle", "Shoulder", "chest"),
        ("arm", "腕", "UpperArm", "UpperArm", f"{_side}_clavicle"),
        ("arm_twist", "腕捩", "UpArmTwist", "", f"{_side}_arm"),
        ("elbow", "ひじ", "Forearm", "LowerArm", f"{_side}_arm"),
        ("wrist_twist", "手捩", "ForeTwist", "", f"{_side}_elbow"),
        ("hand", "手首", "Hand", "Hand", f"{_side}_elbow"),
        ("thigh", "足", "Thigh", "UpperLeg", "pelvis"),
        ("knee", "ひざ", "Calf", "LowerLeg", f"{_side}_thigh"),
        ("foot", "足首", "Foot", "Foot", f"{_side}_knee"),
        ("toe", "つま先", "Toe0", "Toes", f"{_side}_foot"),
    ):
        _key = f"{_side}_{_role}"
        SOURCE_NAMES[_key] = (_jp + _mmd,)
        TARGET_NAMES[_key] = tuple(n for n in (
            f"Bip001_{_side}_{_bip}", f"Bone-{_side}-{_bip}",
            f"Bip001-{_side}-{_bip}", _jp + _mmd,
            ("Left" if _side == "L" else "Right") + _english if _english else "",
        ) if n)
        if _role in {"arm_twist", "wrist_twist"}:
            TARGET_NAMES[_key] = (f"Bip001_{_side}{_bip}",) + TARGET_NAMES[_key]
        if _role in {"thigh", "knee", "foot"}:
            SOURCE_NAMES[_key + "_d"] = (_jp + _mmd + "D",)
        PARENTS[_key] = _parent
    SOURCE_NAMES[f"{_side}_foot_ik"] = (_jp + "足ＩＫ", _jp + "足IK")
    SOURCE_NAMES[f"{_side}_toe_ik"] = (_jp + "つま先ＩＫ", _jp + "つま先IK")
    SOURCE_NAMES[f"{_side}_ik_parent"] = (_jp + "足IK親",)
    SOURCE_NAMES[f"{_side}_shoulder_p"] = (_jp + "肩P",)
    for _number, _finger in enumerate(("親指", "人指", "中指", "薬指", "小指")):
        for _segment in range(3):
            _mmd_segment = _segment if _number == 0 else _segment + 1
            _key = f"{_side}_finger{_number}_{_segment}"
            SOURCE_NAMES[_key] = (_jp + _finger + str(_mmd_segment),)
            _suffix = str(_number) + (str(_segment) if _segment else "")
            TARGET_NAMES[_key] = (f"Bip001_{_side}_Finger{_suffix}", f"Bip001-{_side}-Finger{_suffix}", _jp + _finger + str(_mmd_segment))
            PARENTS[_key] = f"{_side}_hand" if _segment == 0 else f"{_side}_finger{_number}_{_segment - 1}"


def _mmd_module():
    for module in bpy.context.preferences.addons.keys():
        if module == "mmd_tools" or module.endswith(".mmd_tools"):
            return module
    raise ValueError("Enable MMD Tools in Blender Preferences before importing VMD/VPD files.")


def load_motion(filepath):
    """Decode once, then collect compact key arrays (last duplicate key wins)."""
    path = Path(bpy.path.abspath(filepath))
    if not path.is_file() or path.suffix.lower() not in {".vmd", ".vpd"}:
        raise ValueError("Choose an existing .vmd or .vpd file.")
    is_pose = path.suffix.lower() == ".vpd"
    module = importlib.import_module(_mmd_module() + ".core." + ("vpd" if is_pose else "vmd"))
    file = module.File()
    try:
        file.load(filepath=str(path))
    except Exception as exc:
        raise ValueError(f"Cannot read {path.name}: {exc}") from exc
    tracks = {}
    source = ((b.bone_name, [(0, b.location, b.rotation, None)]) for b in file.bones) if is_pose else (
        (name, [(k.frame_number, k.location, k.rotation, k.interp) for k in keys])
        for name, keys in file.boneAnimation.items()
    )
    for name, keys in source:
        keys = [v for _, v in sorted({k[0]: k for k in keys}.items())]
        if not keys:
            continue
        frames = np.asarray([k[0] for k in keys], dtype=np.float64)
        loc = np.asarray([k[1] for k in keys], dtype=np.float64)
        xyzw = np.asarray([k[2] for k in keys], dtype=np.float64)
        if loc.shape != (len(keys), 3) or xyzw.shape != (len(keys), 4) or not np.isfinite(loc).all() or not np.isfinite(xyzw).all():
            raise ValueError(f"Invalid transforms in MMD bone '{name}'.")
        quat = xyzw[:, [3, 0, 1, 2]]
        quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
        if (np.linalg.norm(quat, axis=1) < .5).any():
            raise ValueError(f"Invalid quaternion in MMD bone '{name}'.")
        interp = None if is_pose else np.asarray([k[3] for k in keys], dtype=np.float64)
        tracks[_name(name)] = {"name": name, "frames": frames, "loc": loc, "quat": quat, "interp": interp}
    if not tracks:
        raise ValueError("The file contains no bone motion (camera-only VMD files are not supported).")
    toggles = {}
    if not is_pose:
        for key in file.propertyAnimation or ():
            for name, enabled in key.ik_states:
                toggles.setdefault(_name(name), []).append((key.frame_number, bool(enabled)))
    morph_names = [m.morph_name for m in file.morphs] if is_pose else list(file.shapeKeyAnimation or ())
    return {"tracks": tracks, "toggles": toggles, "pose": is_pose, "path": str(path), "morph_names": morph_names}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _slerp(a, b, t):
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.maximum(np.sin(angle), 1e-12)
    t = np.asarray(t)[..., None]
    q = np.sin((1 - t) * angle) / sine * a + np.sin(t * angle) / sine * b
    return _unit(np.where(dot > .9995, (1 - t) * a + t * b, q))


def _qmatrix(q):
    w, x, y, z = np.moveaxis(_unit(q), -1, 0)
    return np.stack((1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1] + (3, 3))


def _matrix_q(m):
    """Vectorized matrix -> quaternion, including 180 degree rotations."""
    candidates = np.stack((1+m[:,0,0]+m[:,1,1]+m[:,2,2],
                           1+m[:,0,0]-m[:,1,1]-m[:,2,2],
                           1-m[:,0,0]+m[:,1,1]-m[:,2,2],
                           1-m[:,0,0]-m[:,1,1]+m[:,2,2]), axis=1)
    choice = np.argmax(candidates, axis=1)
    q = np.zeros((len(m), 4))
    for i in range(4):
        mask = choice == i
        a = m[mask]
        s = 2 * np.sqrt(np.maximum(candidates[mask, i], 1e-12))
        if i == 0:
            q[mask] = np.stack((s/4, (a[:,2,1]-a[:,1,2])/s, (a[:,0,2]-a[:,2,0])/s, (a[:,1,0]-a[:,0,1])/s), axis=1)
        else:
            j, k = ((2,3), (1,3), (1,2))[i-1]
            q[mask, i] = s/4
            q[mask, 0] = (a[:, k-1, j-1]-a[:, j-1, k-1])/s * (1 if i != 2 else -1)
            q[mask, j] = (a[:, i-1, j-1]+a[:, j-1, i-1])/s
            q[mask, k] = (a[:, i-1, k-1]+a[:, k-1, i-1])/s
    q = _unit(q)
    flips = np.sum(q[1:] * q[:-1], axis=1) < 0
    signs = np.r_[1, np.cumprod(np.where(flips, -1, 1))]
    return q * signs[:, None]


def _bezier(t, control):
    x1, y1, x2, y2 = np.moveaxis(np.clip(control / 127, 0, 1), -1, 0)
    lo, hi = np.zeros_like(t), np.ones_like(t)
    for _ in range(18):
        u = (lo + hi) * .5
        x = 3*(1-u)**2*u*x1 + 3*(1-u)*u*u*x2 + u**3
        lo, hi = np.where(x < t, u, lo), np.where(x < t, hi, u)
    u = (lo + hi) * .5
    return 3*(1-u)**2*u*y1 + 3*(1-u)*u*u*y2 + u**3


def sample_track(track, times):
    frames = track["frames"]
    hi = np.clip(np.searchsorted(frames, times, side="right"), 0, len(frames)-1)
    lo = np.maximum(hi-1, 0)
    lo = np.where(times >= frames[-1], hi, lo)
    t = np.clip((times-frames[lo]) / np.maximum(frames[hi]-frames[lo], 1e-12), 0, 1)
    amounts = np.repeat(t[:, None], 4, axis=1)
    if track["interp"] is not None:
        for axis in range(4):
            control = track["interp"][hi][:, [axis, axis+4, axis+8, axis+12]]
            amounts[:, axis] = _bezier(t, control)
    loc = track["loc"][lo] + amounts[:, :3] * (track["loc"][hi]-track["loc"][lo])
    q = _slerp(track["quat"][lo], track["quat"][hi], amounts[:, 3])
    return loc, q


def _rotate_between(a, b):
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    q = np.concatenate((1 + np.sum(a*b, axis=-1, keepdims=True), axis), axis=-1)
    reverse = q[:, 0] < 1e-8
    helper = np.broadcast_to(np.array([1., 0, 0]), a.shape).copy()
    helper[np.abs(a[:, 0]) > .9] = (0, 1, 0)
    q[reverse, 0] = 0
    q[reverse, 1:] = _unit(np.cross(a[reverse], helper[reverse]))
    return _qmatrix(_unit(q))


def _mapping(rig):
    lookup = {}
    for bone in rig.pose.bones:
        lookup.setdefault(_target_name(bone.name), []).append(bone.name)
        mmd = getattr(bone, "mmd_bone", None)
        if mmd and mmd.name_j:
            lookup.setdefault(_target_name(mmd.name_j), []).append(bone.name)
    mapping = {}
    for role, aliases in TARGET_NAMES.items():
        for alias in aliases:
            if alias in rig.data.bones:
                mapping[role] = alias
                break
            found = set(lookup.get(_target_name(alias), ()))
            if len(found) > 1:
                raise ValueError(f"Ambiguous target bone for '{alias}': {', '.join(sorted(found))}.")
            if found:
                mapping[role] = found.pop()
                break
    if "pelvis" not in mapping or not ({"spine", "head", "L_thigh", "R_thigh"} & mapping.keys()):
        raise ValueError("Target bone names are not a supported MMD / Bip001 / humanoid body skeleton.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Two MMD body roles resolve to the same target bone.")
    return mapping


def _source_tracks(motion):
    found = {}
    for role, aliases in SOURCE_NAMES.items():
        for alias in aliases:
            if _name(alias) in motion["tracks"]:
                found[role] = motion["tracks"][_name(alias)]
                break
    # Models without the optional thumb-0 bone start at thumb-1.
    for side, jp in (("L", "左"), ("R", "右")):
        if _name(jp + "親指0") not in motion["tracks"]:
            for segment in range(3):
                key = f"{side}_finger0_{segment}"
                track = motion["tracks"].get(_name(jp + "親指" + str(segment + 1)))
                found.pop(key, None)
                if track:
                    found[key] = track
    return found


def _ik_enabled(motion, track, times):
    keys = motion["toggles"].get(_name(track["name"]), ())
    if not keys:
        return np.ones(len(times), dtype=bool)
    keys = sorted(dict(keys).items())
    frames, values = np.asarray([k[0] for k in keys]), np.asarray([k[1] for k in keys])
    indices = np.searchsorted(frames, times, side="right")-1
    return np.where(indices < 0, True, values[np.maximum(indices, 0)])


def _fold_controls(controls, identity):
    """EIEM pre/base/post composition; D controls never replace base tracks."""
    rotations = {role: value[1] for role, value in controls.items()}

    def fold(role, pre=None, post=None):
        rotations[role] = (identity if pre is None else pre) @ rotations.get(role, identity) @ (identity if post is None else post)

    for side in ('L', 'R'):
        for role in ('thigh', 'knee', 'foot'):
            key = f'{side}_{role}'
            if key + '_d' in rotations:
                fold(key, rotations[key + '_d'])
        shoulder = rotations.get(f'{side}_shoulder_p')
        if shoulder is not None:
            fold(f'{side}_clavicle', shoulder)
            fold(f'{side}_arm', np.swapaxes(shoulder, -1, -2))
    if 'waist' in rotations:
        waist = rotations['waist']
        fold('spine', waist)
        fold('pelvis', waist)
        for side in ('L', 'R'):
            fold(f'{side}_thigh', np.swapaxes(waist, -1, -2))
    fold('chest', rotations.get('chest_pre'), rotations.get('chest_extra'))
    return rotations


def _frame(forward, lateral):
    forward = _unit(forward)
    lateral = lateral - forward * np.dot(forward, lateral)
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError('Degenerate wrist reference frame.')
    lateral = _unit(lateral)
    normal = _unit(np.cross(forward, lateral))
    return np.column_stack((np.cross(normal, forward), normal, forward))


def _stance_corrections(heads, reference):
    """Return target -> source bind rotations (EIEM inverse alignment)."""
    children = {'spine': 'chest', 'chest': 'neck', 'neck': 'head'}
    for side in ('L', 'R'):
        children.update({f'{side}_clavicle': f'{side}_arm',
                         f'{side}_arm': f'{side}_elbow', f'{side}_elbow': f'{side}_hand'})
    corrections = {}
    for role, child in children.items():
        if role not in heads or child not in heads:
            continue
        direction = heads[child] - heads[role]
        if np.linalg.norm(direction) < 1e-6:
            raise ValueError(f'{role} and {child} occupy the same position.')
        source = reference['directions'][role]
        corrections[role] = _rotate_between(direction[None], source[None])[0]
    for side in ('L', 'R'):
        hand = f'{side}_hand'
        if hand not in heads:
            continue
        middle, index, little = (f'{side}_finger{n}_0' for n in (2, 1, 4))
        if all(role in heads for role in (middle, index, little)):
            source = _frame(*reference['wrists'][side])
            target = _frame(heads[middle]-heads[hand], heads[index]-heads[little])
            corrections[hand] = source @ target.T
        elif f'{side}_elbow' in heads:
            # Without palm landmarks retain forearm roll, rather than invent it.
            corrections[hand] = corrections.get(f'{side}_elbow', np.eye(3))
        for role in heads:
            if role.startswith(f'{side}_finger'):
                corrections[role] = corrections.get(hand, np.eye(3))
    return corrections


def _twist_angle(rotation, axis):
    """Signed swing-twist projection around the SOURCE reference axis."""
    q = _matrix_q(rotation)
    projected = q[:, 1:] @ _unit(axis)
    valid = np.hypot(projected, q[:, 0]) > 1e-6
    angle = 2 * np.arctan2(projected, q[:, 0])
    return np.where(valid, (angle + np.pi) % (2*np.pi) - np.pi, 0.)


def _axis_rotation(axis, angle):
    axis = _unit(np.asarray(axis))
    return _qmatrix(np.column_stack((np.cos(angle*.5), np.sin(angle*.5)[:, None]*axis)))


def _ramp(value, lo, hi):
    t = np.clip((value-lo)/(hi-lo), 0, 1)
    return t*t*(3-2*t)


def _mix_knee(axis, pole, source_upper, source_lower, distance, reach, effective_fk):
    """EIEM confidence-weighted FK knee hint, guarded near full extension."""
    u, l = _unit(source_upper), _unit(source_lower)
    normal = np.cross(u, l)
    confidence = _ramp(np.linalg.norm(normal, axis=1), .035, .174) * effective_fk
    source = _unit(np.cross(_unit(u+l), normal))
    source -= axis*np.sum(source*axis, axis=1, keepdims=True)
    projection = np.linalg.norm(source, axis=1)
    source = _unit(source)
    dot = np.clip(np.sum(pole*source, axis=1), -1, 1)
    weight = .35 * confidence * _ramp(dot, -.98, -.8) * (1-_ramp(distance/reach, .995, .99995)) * _ramp(projection, .035, .174)
    angle = np.arctan2(np.sum(axis*np.cross(pole, source), axis=1), dot)*weight
    return pole*np.cos(angle)[:, None] + np.cross(axis, pole)*np.sin(angle)[:, None]


def _target_order(rig, mapping):
    """Collect target ancestors and known deform twist chains once per import."""
    helpers = []
    for side in ('L', 'R'):
        for role, end in (('arm', 'elbow'), ('elbow', 'hand')):
            first_role = f'{side}_' + ('arm_twist' if role == 'arm' else 'wrist_twist')
            first = mapping.get(first_role)
            if not first:
                continue
            second = next((n for n in (first+'1',) if n in rig.data.bones), None)
            if second and rig.data.bones[second].parent != rig.data.bones[first]:
                raise ValueError(f"'{second}' is not a child of '{first}'; unsupported twist chain.")
            base = mapping.get(f'{side}_{role}')
            end_name = mapping.get(f'{side}_{end}')
            if not base or not end_name or rig.data.bones[first].parent != rig.data.bones[base]:
                raise ValueError(f"'{first}' is not a separate deform branch of the limb.")
            helpers.append((first, second, f'{side}_{role}', f'{side}_{end}', first_role))
    names = set(mapping.values())
    names.update(n for h in helpers for n in h[:2] if n)
    for name in tuple(names):
        parent = rig.data.bones[name].parent
        while parent and parent.name not in names:
            names.add(parent.name)
            parent = parent.parent
    # data.bones are not guaranteed to be in parent order after editing.
    ordered, visited = [], set()
    def visit(bone):
        if bone.name in visited:
            return
        if bone.parent:
            visit(bone.parent)
        visited.add(bone.name)
        ordered.append(bone)
    for name in sorted(names):
        visit(rig.data.bones[name])
    for bone in ordered:
        if bone.inherit_scale != 'FULL' or not bone.use_inherit_rotation or not bone.use_local_location:
            raise ValueError(f"'{bone.name}' uses nonstandard transform inheritance; use an FK/game skeleton.")
        if any(not c.mute and c.influence > 0 for c in rig.pose.bones[bone.name].constraints):
            raise ValueError(f"'{bone.name}' has active constraints; import onto an FK/game skeleton.")
    return ordered, helpers


def _target_fk(ordered, rest_heads, desired, root_positions, count):
    """Rotation-only FK: keep every original bone offset and length."""
    identity = np.broadcast_to(np.eye(3), (count, 3, 3))
    positions, rotations = {}, {}
    for bone in ordered:
        name = bone.name
        inherited = rotations[bone.parent.name] if bone.parent else identity
        rotations[name] = desired.get(name, inherited)
        if name in root_positions:
            positions[name] = root_positions[name]
        elif bone.parent:
            positions[name] = positions[bone.parent.name] + np.einsum('nij,j->ni', inherited, rest_heads[name]-rest_heads[bone.parent.name])
        else:
            positions[name] = np.broadcast_to(rest_heads[name], (count, 3)).copy()
    return positions, rotations


def _solve_legs(mapping, heads, controls, source_world, tracks, motion, times,
                reference, base_scale, virtual_root, positions, rotations, desired):
    count = len(times)
    zero, identity = np.zeros((count, 3)), np.broadcast_to(np.eye(3), (count, 3, 3))
    root_pos, root_rot, root_head = virtual_root
    clipped = 0
    for side in ('L', 'R'):
        hip, knee, foot, toe = (f'{side}_{part}' for part in ('thigh', 'knee', 'foot', 'toe'))
        ik = f'{side}_foot_ik'
        if not all(role in mapping for role in (hip, knee, foot)):
            continue
        enabled = _ik_enabled(motion, tracks.get(ik, {'name': SOURCE_NAMES[ik][0]}), times)
        ik_loc, ik_rot = controls.get(ik, (zero, identity))
        par_loc, par_rot = controls.get(f'{side}_ik_parent', (zero, identity))
        pivot_offset = reference['foot_from_parent']*base_scale
        parent_head = heads[foot] - pivot_offset
        goal_local = parent_head-root_head+par_loc + np.einsum('nij,nj->ni', par_rot, pivot_offset+ik_loc)
        goal = root_pos + np.einsum('nij,nj->ni', root_rot, goal_local)
        hip_name, knee_name, foot_name = (mapping[role] for role in (hip, knee, foot))
        h, k, f = (positions[name] for name in (hip_name, knee_name, foot_name))
        a, b = np.linalg.norm(heads[knee]-heads[hip]), np.linalg.norm(heads[foot]-heads[knee])
        if min(a, b) < 1e-6:
            raise ValueError(f'{side} leg joints occupy the same position.')
        delta = goal-h
        distance = np.linalg.norm(delta, axis=1)
        clipped += int((enabled & ((distance > a+b+1e-5) | (distance < abs(a-b)-1e-5))).sum())
        d = np.clip(distance, abs(a-b)+1e-7, a+b-1e-7)
        axis = _unit(delta)
        axis[distance < 1e-9] = (0, 0, -1)
        along = (a*a-b*b+d*d)/(2*d)
        height = np.sqrt(np.maximum(a*a-along*along, 0))
        # Consistent MMD knee direction. Mix source FK only when it bends enough
        # to carry a reliable plane; never flip an almost straight knee.
        pelvis_rot = source_world['pelvis']
        pole = np.einsum('nij,j->ni', pelvis_rot, np.array([0., -1., 0.]))
        pole -= axis*np.sum(pole*axis, axis=1, keepdims=True)
        degenerate = np.linalg.norm(pole, axis=1) < 1e-6
        alternative = np.cross(axis, np.array([1., 0., 0.]))
        alternative[np.linalg.norm(alternative, axis=1) < 1e-6] = (0, 1, 0)
        pole = _unit(np.where(degenerate[:, None], alternative, pole))
        source_upper = np.einsum('nij,j->ni', source_world[hip], heads[knee]-heads[hip])
        source_lower = np.einsum('nij,j->ni', source_world[knee], heads[foot]-heads[knee])
        knee_local = source_world[hip].transpose(0, 2, 1) @ source_world[knee]
        effective_fk = np.linalg.norm(knee_local-identity, axis=(1, 2)) > 1e-5
        pole = _mix_knee(axis, pole, source_upper, source_lower, distance, a+b, effective_fk)
        new_knee = h + axis*along[:, None] + pole*height[:, None]
        new_foot = h + axis*d[:, None]
        desired[hip_name] = np.where(enabled[:, None, None], _rotate_between(k-h, new_knee-h) @ rotations[hip_name], rotations[hip_name])
        desired[knee_name] = np.where(enabled[:, None, None], _rotate_between(f-k, new_foot-new_knee) @ rotations[knee_name], rotations[knee_name])
        # IK positionWeight=1, rotationWeight=0, then minimal ankle toe aim.
        # Foot IK rotation rotates the TOE TARGET; it is not the ankle rotation.
        foot_rotation = rotations[foot_name]
        toe_ik = f'{side}_toe_ik'
        if toe in mapping:
            toe_enabled = enabled & _ik_enabled(motion, tracks.get(toe_ik, {'name': SOURCE_NAMES[toe_ik][0]}), times)
            ik_world = root_rot @ par_rot @ ik_rot
            toe_goal = goal + np.einsum('nij,nj->ni', ik_world, heads[toe]-heads[foot]+controls.get(toe_ik, (zero, identity))[0])
            current = np.einsum('nij,j->ni', foot_rotation, heads[toe]-heads[foot])
            actual_foot = np.where(enabled[:, None], new_foot, f)
            aim = _rotate_between(current, toe_goal-actual_foot) @ foot_rotation
            foot_rotation = np.where(toe_enabled[:, None, None], aim, foot_rotation)
        desired[foot_name] = foot_rotation
        if toe in mapping:
            toe_local = source_world[foot].transpose(0, 2, 1) @ source_world[toe]
            desired[mapping[toe]] = foot_rotation @ toe_local
    return clipped


def retarget_motion(rig, motion, *, translation_multiplier=1., start_frame=1, fps=30):
    """EIEM reference -> calibrated FK -> leg/toe IK -> deform twists -> bake.

    Computes armature-local samples only. Source controls use the documented
    Tda reference; VMD/VPD do not contain a source PMX bind/constraint graph.
    """
    if not is_armature_object(rig):
        raise ValueError('Choose a target armature.')
    if rig.data.pose_position != 'POSE' or rig.library or rig.data.library:
        raise ValueError('Choose an editable armature in Pose Position.')
    if not math.isfinite(translation_multiplier) or translation_multiplier <= 0:
        raise ValueError('Translation multiplier must be positive.')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Scene frame rate must be positive.')
    mapping, tracks = _mapping(rig), _source_tracks(motion)
    if not tracks:
        raise ValueError('No standard MMD body controls in this file.')
    usable = set(mapping) | {'root', 'center', 'groove', 'waist'}
    if 'chest' in mapping:
        usable.update(('chest_pre', 'chest_extra'))
    for side in ('L', 'R'):
        if f'{side}_clavicle' in mapping:
            usable.add(f'{side}_shoulder_p')
        for role in ('thigh', 'knee', 'foot'):
            if f'{side}_{role}' in mapping:
                usable.add(f'{side}_{role}_d')
        if all(f'{side}_{role}' in mapping for role in ('thigh', 'knee', 'foot')):
            usable.update((f'{side}_foot_ik', f'{side}_ik_parent'))
            if f'{side}_toe' in mapping:
                usable.add(f'{side}_toe_ik')
    if not (usable & tracks.keys()):
        raise ValueError('No MMD controls in this file match the target skeleton.')
    if rig.animation_data:
        if any(not t.mute for t in rig.animation_data.nla_tracks):
            raise ValueError("Mute the target's NLA tracks before importing.")
        if any(d.data_path.startswith('pose.bones[') for d in rig.animation_data.drivers):
            raise ValueError('The target has pose drivers; use an FK/game skeleton.')
    ordered, helpers = _target_order(rig, mapping)
    end = max(float(t['frames'][-1]) for t in tracks.values())
    count = 1 if motion['pose'] else int(math.ceil(end*fps/30))+1
    if count > 60000 or count*len(ordered) > 2500000:
        raise ValueError('This motion is too large for one import; split it into shorter clips.')
    frames = start_frame + np.arange(count, dtype=np.float64)
    times = np.minimum(np.arange(count)*30/fps, end)
    identity = np.broadcast_to(np.eye(3), (count, 3, 3))
    zero = np.zeros((count, 3))
    rest_heads = {b.name: np.asarray(b.head_local, dtype=np.float64) for b in ordered}
    heads = {role: rest_heads[name] for role, name in mapping.items()}
    reference = tda_reference()
    lengths = [np.linalg.norm(heads[f'{s}_knee']-heads[f'{s}_thigh']) + np.linalg.norm(heads[f'{s}_foot']-heads[f'{s}_knee'])
               for s in ('L', 'R') if all(f'{s}_{part}' in heads for part in ('thigh', 'knee', 'foot'))]
    if not lengths or min(lengths) < 1e-6:
        raise ValueError('A complete thigh/knee/foot chain is required for automatic MMD reference scaling.')
    base_scale = float(np.mean(lengths))/reference['leg_length']
    scale = base_scale*translation_multiplier
    swap = MMD_TO_BLENDER
    controls = {}
    for role, track in tracks.items():
        loc, q = sample_track(track, times)
        controls[role] = (loc @ swap.T*scale, swap @ _qmatrix(q) @ swap.T)
    local_rotations = _fold_controls(controls, identity)
    source_world = {}
    for role, parent in PARENTS.items():
        local = local_rotations.get(role, identity)
        source_world[role] = local if parent is None else source_world[parent] @ local
    correction = _stance_corrections(heads, reference)
    desired = {name: source_world[role] @ correction.get(role, np.eye(3))
               for role, name in mapping.items() if role in source_world and not role.endswith(('arm_twist', 'wrist_twist'))}
    root_head = heads.get('root', np.zeros(3))
    center_head = heads['pelvis'] - reference['lower_from_center']*base_scale
    groove_head = center_head + reference['groove_from_center']*base_scale
    virtual_heads = {'root': root_head, 'center': center_head, 'groove': groove_head}
    virtual_pos = {}
    for role, parent in (('root', None), ('center', 'root'), ('groove', 'center')):
        loc = controls.get(role, (zero, identity))[0]
        virtual_pos[role] = virtual_heads[role] + loc if parent is None else virtual_pos[parent] + np.einsum('nij,nj->ni', source_world[parent], virtual_heads[role]-virtual_heads[parent]+loc)
    # Move the existing root about the virtual Groove pivot. All downstream
    # local offsets stay at rest: no separately translated elbow/ankle joints.
    root_role = 'root' if 'root' in mapping else 'pelvis'
    root_name = mapping[root_role]
    groove_rot = source_world['groove']
    root_positions = {root_name: virtual_pos['groove'] + np.einsum('nij,j->ni', groove_rot, heads[root_role]-groove_head)}
    if root_role == 'root':
        desired[root_name] = groove_rot
    positions, rotations = _target_fk(ordered, rest_heads, desired, root_positions, count)
    clipped = _solve_legs(mapping, heads, controls, source_world, tracks, motion, times,
                          reference, base_scale, (virtual_pos['root'], source_world['root'], root_head),
                          positions, rotations, desired)
    # Populate helper branches even when the clip contains no explicit twist:
    # their FK inheritance must not retain a pose from the previous action.
    for first, second, base, end_role, twist_role in helpers:
        angle = _twist_angle(controls[twist_role][1], reference['directions'][base]) if twist_role in controls else np.zeros(count)
        axis = heads[end_role]-heads[base]
        if np.linalg.norm(axis) < 1e-6:
            raise ValueError(f'{base} has a degenerate twist axis.')
        base_rotation = desired[mapping[base]]
        desired[first] = base_rotation @ _axis_rotation(axis, angle*(.5 if second else 1.))
        if second:
            desired[second] = base_rotation @ _axis_rotation(axis, angle)
    positions, rotations = _target_fk(ordered, rest_heads, desired, root_positions, count)
    samples = {}
    for bone in ordered:
        name = bone.name
        rest_rot = np.asarray(bone.matrix_local, dtype=np.float64)[:3, :3]
        parent_rot = rotations[bone.parent.name] if bone.parent else identity
        basis_rot = rest_rot.T @ parent_rot.transpose(0, 2, 1) @ rotations[name] @ rest_rot
        loc = zero.copy()
        if name in root_positions:
            inherited_pos = positions[bone.parent.name] + np.einsum('nij,j->ni', parent_rot, rest_heads[name]-rest_heads[bone.parent.name]) if bone.parent else np.broadcast_to(rest_heads[name], (count, 3))
            loc = np.einsum('nij,nj->ni', (parent_rot @ rest_rot).transpose(0, 2, 1), positions[name]-inherited_pos)
        samples[name] = (loc, _matrix_q(basis_rot))
    used = {_name(t['name']) for t in tracks.values()}
    ignored = [t['name'] for n, t in motion['tracks'].items() if n not in used]
    missing = [tracks[r]['name'] for r in tracks if r in PARENTS and r not in mapping and r not in {'root', 'center', 'groove'}]
    translation_roles = {'root', 'center', 'groove', 'L_foot_ik', 'R_foot_ik', 'L_ik_parent', 'R_ik_parent', 'L_toe_ik', 'R_toe_ik'}
    untranslated = [tracks[r]['name'] for r in controls if r not in translation_roles and np.max(np.abs(controls[r][0])) > 1e-6]
    for side in ('L', 'R'):
        for suffix in ('arm_twist', 'wrist_twist'):
            role = f'{side}_{suffix}'
            if role in tracks and role not in mapping:
                missing.append(tracks[role]['name'])
        for suffix in ('foot_ik', 'toe_ik'):
            role = f'{side}_{suffix}'
            required = ('thigh', 'knee', 'foot') + (('toe',) if suffix == 'toe_ik' else ())
            if role in tracks and not all(f'{side}_{r}' in mapping for r in required):
                missing.append(tracks[role]['name'])
    if not all(np.isfinite(a).all() for pair in samples.values() for a in pair):
        raise ValueError('Non-finite retargeted transforms; nothing was changed.')
    return {'frames': frames, 'samples': samples, 'mapping': mapping, 'ignored': ignored,
            'missing': missing, 'untranslated': untranslated, 'ik_clipped_samples': clipped,
            'morph_names': motion['morph_names'], 'pose': motion['pose'],
            'reference': reference['name'], 'translation_scale': scale}


def _action_curves(action, rig):
    if bpy.app.version >= (4, 4, 0):
        slot = action.slots.new(id_type="OBJECT", name=rig.name)
        strip = action.layers.new("MMD Body Motion").strips.new(type="KEYFRAME")
        return strip.channelbags.new(slot).fcurves, slot
    return action.fcurves, None


def import_mmd_motion(context, rig, filepath, *, translation_multiplier=1., start_frame=1):
    motion = load_motion(filepath)
    fps = context.scene.render.fps / context.scene.render.fps_base
    result = retarget_motion(rig, motion, translation_multiplier=translation_multiplier, start_frame=start_frame, fps=fps)
    samples, frames = result["samples"], result["frames"]
    action = bpy.data.actions.new(Path(motion["path"]).stem + " -> " + rig.name)
    previous = rig.animation_data.action if rig.animation_data else None
    previous_fake_user = previous.use_fake_user if previous else False
    previous_slot = getattr(rig.animation_data, "action_slot", None) if rig.animation_data else None
    before = {n: (rig.pose.bones[n].rotation_mode, rig.pose.bones[n].matrix_basis.copy()) for n in samples}
    old_frame, old_end = context.scene.frame_current, context.scene.frame_end
    had_animation_data = rig.animation_data is not None
    try:
        curves, slot = _action_curves(action, rig)
        linear = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value
        for name, (locations, quaternions) in samples.items():
            pb = rig.pose.bones[name]
            for prop, values in (("location", locations), ("rotation_quaternion", quaternions), ("scale", np.ones_like(locations))):
                path = pb.path_from_id(prop)
                # Constant channels need only endpoints; dense channels use bulk writes.
                constant = np.max(np.abs(values-values[:1])) < 1e-7
                indices = np.array([0]) if len(frames) == 1 else np.array([0,len(frames)-1]) if constant else np.arange(len(frames))
                for axis in range(values.shape[1]):
                    fc = curves.new(data_path=path, index=axis)
                    fc.keyframe_points.add(len(indices))
                    coordinates = np.column_stack((frames[indices], values[indices,axis])).astype(np.float32)
                    fc.keyframe_points.foreach_set("co", coordinates.ravel())
                    fc.keyframe_points.foreach_set("interpolation", np.full(len(indices), linear, dtype=np.int32))
                    fc.update()
        action["mdt_mmd_source"] = motion["path"]
        action["mdt_mmd_reference"] = result["reference"]
        action["mdt_mmd_translation_scale"] = result["translation_scale"]
        action["mdt_mmd_report"] = "; ".join(filter(None, (
            "Missing target roles: " + ", ".join(result["missing"]) if result["missing"] else "",
            "Ignored controls: " + ", ".join(result["ignored"]) if result["ignored"] else "",
            "Unsupported local translations: " + ", ".join(result["untranslated"]) if result["untranslated"] else "",
            "Unmapped morphs: " + ", ".join(result["morph_names"]) if result["morph_names"] else "",
            f"Clamped unreachable leg IK samples: {result['ik_clipped_samples']}" if result["ik_clipped_samples"] else "",
        )))
        rig.animation_data_create().action = action
        if slot is not None:
            rig.animation_data.action_slot = slot
        for name in samples:
            pb = rig.pose.bones[name]
            pb.rotation_mode = "QUATERNION"
            pb.scale = (1,1,1)
        # Keep the old clip accessible even if it no longer has an object user.
        if previous:
            previous.use_fake_user = True
        action.use_fake_user = True
        context.scene.frame_end = max(old_end, int(frames[-1]))
        context.scene.frame_set(start_frame)
        result["action"] = action.name
        return result
    except Exception:
        if previous:
            previous.use_fake_user = previous_fake_user
        if rig.animation_data:
            rig.animation_data.action = previous
            if previous_slot is not None:
                rig.animation_data.action_slot = previous_slot
        for name, (mode, basis) in before.items():
            pb = rig.pose.bones[name]
            pb.rotation_mode = mode
            pb.matrix_basis = basis
        context.scene.frame_end = old_end
        context.scene.frame_set(old_frame)
        if not had_animation_data:
            rig.animation_data_clear()
        bpy.data.actions.remove(action)
        raise
