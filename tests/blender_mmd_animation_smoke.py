"""Background tests: coordinates, FK/IK, interpolation, time and action safety."""

import importlib
import math
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
extensions = Path(bpy.utils.user_resource("EXTENSIONS")) / "blender_org"
sys.path.insert(0, str(extensions))
if "mmd_tools" not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module="mmd_tools")

from modtool.core.mmd_animation import (import_mmd_motion, load_motion, retarget_motion,
                                      _matrix_q, _qmatrix, sample_track)


def make_rig():
    data = bpy.data.armatures.new("MMDTestData")
    rig = bpy.data.objects.new("MMDTest", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    entries = [
        ("Root", (0,0,0), None), ("root", (0,0,.01), "Root"),
        ("Bip001_Pelvis", (0,0,1), "Root"),
        ("Bip001_Spine", (0,0,1.1), "Bip001_Pelvis"),
        ("Bip001_Spine1", (0,0,1.2), "Bip001_Spine"),
        ("Bip001_Spine2", (0,0,1.3), "Bip001_Spine1"),
        ("Bip001_Neck", (0,0,1.5), "Bip001_Spine2"),
        ("Bip001_Head", (0,0,1.6), "Bip001_Neck"),
    ]
    for side, sign in (("L",1), ("R",-1)):
        entries += [
            (f"Bip001_{side}_Clavicle", (.05*sign,0,1.4), "Bip001_Spine2"),
            (f"Bip001_{side}_UpperArm", (.12*sign,0,1.4), f"Bip001_{side}_Clavicle"),
            (f"Bip001_{side}_Forearm", (.35*sign,0,1.2), f"Bip001_{side}_UpperArm"),
            (f"Bip001_{side}_Hand", (.55*sign,0,1.0), f"Bip001_{side}_Forearm"),
            (f"Bip001_{side}_Thigh", (.08*sign,0,1), "Bip001_Pelvis"),
            (f"Bip001_{side}_Calf", (.08*sign,-.01,.55), f"Bip001_{side}_Thigh"),
            (f"Bip001_{side}_Foot", (.08*sign,0,.10), f"Bip001_{side}_Calf"),
            (f"Bip001_{side}_Toe0", (.08*sign,-.12,.05), f"Bip001_{side}_Foot"),
        ]
    for index, (name, head, parent) in enumerate(entries):
        bone = data.edit_bones.new(name)
        bone.head = head
        bone.tail = Vector(head)+Vector((.03,.04,.02))
        bone.roll = (index % 4)*.4
        if parent:
            bone.parent = data.edit_bones[parent]
    for side in ("L", "R"):
        for stem, base, end in (("UpArmTwist", "UpperArm", "Forearm"), ("_ForeTwist", "Forearm", "Hand")):
            first = data.edit_bones.new(f"Bip001_{side}{stem}")
            start = data.edit_bones[f"Bip001_{side}_{base}"]
            stop = data.edit_bones[f"Bip001_{side}_{end}"]
            first.head, first.tail = start.head, start.tail
            first.parent = start
            second = data.edit_bones.new(first.name + "1")
            second.head = (start.head + stop.head) * .5
            second.tail = second.head + (start.tail - start.head)
            second.parent = first
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def vpd_file(folder, entries, name="neutral_test.vpd"):
    vpd = importlib.import_module("mmd_tools.core.vpd")
    file = vpd.File()
    file.osm_name = "Test.osm"
    file.bones = [vpd.VpdBone(n, p, q) for n,p,q in entries]
    path = str(Path(folder)/name)
    file.save(filepath=path)
    return path


def curves(action):
    return action.layers[0].strips[0].channelbags[0].fcurves if bpy.app.version >= (4,4,0) else action.fcurves


def deformation(rig, name):
    pb = rig.pose.bones[name]
    return np.asarray(pb.matrix.to_3x3() @ pb.bone.matrix_local.to_3x3().inverted())


def run():
    rig = make_rig()
    identity = (0,0,0,1)
    q45 = (0, math.sin(math.pi/8), 0, math.cos(math.pi/8))
    from modtool.core.mmd_animation import _twist_angle
    from modtool.core.mmd_reference import tda_reference, MMD_TO_BLENDER
    reference = tda_reference()
    before = {b.name: np.asarray(b.matrix_local).copy() for b in rig.data.bones}
    with tempfile.TemporaryDirectory(prefix="modtool_mmd_test_") as folder:
        neutral_path = vpd_file(folder, [("下半身", (0,0,0), identity)])
        neutral = import_mmd_motion(bpy.context, rig, neutral_path)
        baseline = {b.name: deformation(rig, b.name) for b in rig.pose.bones}
        # An independent direction oracle from the EIEM source constants.
        # Tail directions and bone roll in this fixture are deliberately wrong.
        upper, elbow = (rig.pose.bones['Bip001_L_'+p] for p in ('UpperArm', 'Forearm'))
        expected = reference['directions']['L_arm']
        np.testing.assert_allclose((elbow.head-upper.head).normalized(), expected/np.linalg.norm(expected), atol=2e-5)
        old = rig.animation_data.action
        scale = neutral['translation_scale']
        path = vpd_file(folder, [("センター", (1,2,3), identity), ("左腕", (0,0,0), q45)])
        result = import_mmd_motion(bpy.context, rig, path, start_frame=7)
        assert bpy.context.scene.frame_current == 7 and old.use_fake_user
        assert all(np.array_equal(np.asarray(b.matrix_local),before[b.name]) for b in rig.data.bones)
        np.testing.assert_allclose(rig.pose.bones['Bip001_Pelvis'].head, [scale,3*scale,1+2*scale], atol=1e-5)
        expected_rot = np.asarray(Matrix.Rotation(-math.pi/4,3,'Z')) @ baseline['Bip001_L_UpperArm']
        np.testing.assert_allclose(deformation(rig, 'Bip001_L_UpperArm'), expected_rot, atol=2e-5)
        # Ordinary joints never acquire local translation/scale animation.
        assert all(np.linalg.norm(v[0]) < 1e-7 for n,v in result['samples'].items() if n != 'Root')
        assert all(np.max(np.abs(np.asarray(rig.pose.bones[n].scale)-1)) < 1e-6 for n in result['samples'])
        # Lower-body rotation must not leak into the upper body's world rotation.
        path = vpd_file(folder, [("下半身",(0,0,0),q45), ("上半身",(0,0,0),identity)], 'torso.vpd')
        import_mmd_motion(bpy.context, rig, path)
        np.testing.assert_allclose(deformation(rig,'Bip001_Spine'), baseline['Bip001_Spine'], atol=2e-5)
        # Waist and ShoulderP have explicit inverse cancellation, not aliasing.
        path = vpd_file(folder, [("腰",(0,0,0),q45), ("左肩P",(0,0,0),q45)], 'semi.vpd')
        motion = load_motion(path)
        motion['toggles'] = {'左足ik': [(0,False)], '右足ik': [(0,False)]}
        # Use neutral write helper below to exercise the actual Blender matrices.
        def apply_samples(result):
            for name, (loc, q) in result['samples'].items():
                pb=rig.pose.bones[name]
                pb.rotation_mode='QUATERNION'
                pb.location=loc[0]; pb.rotation_quaternion=q[0]; pb.scale=(1,1,1)
            rig.animation_data.action = None
            bpy.context.view_layer.update()
        apply_samples(retarget_motion(rig,motion))
        waist = np.asarray(Matrix.Rotation(-math.pi/4,3,'Z'))
        np.testing.assert_allclose(deformation(rig,'Bip001_L_UpperArm'), waist @ baseline['Bip001_L_UpperArm'],atol=2e-5)
        np.testing.assert_allclose(deformation(rig,'Bip001_L_Thigh'),np.eye(3),atol=2e-5)
        # D rotations compose with the base track in the specified order.
        qx = (math.sin(.2),0,0,math.cos(.2))
        path = vpd_file(folder, [("左足",(0,0,0),q45), ("左足D",(0,0,0),qx)], 'leg_d.vpd')
        motion=load_motion(path); motion['toggles']={'左足ik':[(0,False)],'右足ik':[(0,False)]}
        apply_samples(retarget_motion(rig,motion))
        qx_bl = np.asarray(Matrix.Rotation(-.4,3,'X'))
        np.testing.assert_allclose(deformation(rig,'Bip001_L_Thigh'),qx_bl @ waist,atol=2e-5)
        # Center uses its reference pivot, not the pelvis head.
        path = vpd_file(folder, [("センター",(0,0,0),qx)], 'center_pivot.vpd')
        motion=load_motion(path); motion['toggles']={'左足ik':[(0,False)],'右足ik':[(0,False)]}
        apply_samples(retarget_motion(rig,motion))
        center = np.array([0,0,1.])-reference['lower_from_center']*scale
        expected_hip = center + qx_bl @ (np.array([0,0,1.])-center)
        np.testing.assert_allclose(rig.pose.bones['Bip001_Pelvis'].head,expected_hip,atol=2e-5)
        # Both feet stay planted when Center lowers, even without explicit IK keys.
        ik_path = vpd_file(folder, [("センター",(0,-2,0),identity)], 'ik.vpd')
        import_mmd_motion(bpy.context,rig,ik_path)
        for side, sign in (('L',1),('R',-1)):
            np.testing.assert_allclose(rig.pose.bones[f'Bip001_{side}_Foot'].head,[sign*.08,0,.1],atol=2e-5)
            assert rig.pose.bones[f'Bip001_{side}_Calf'].head.y < -.1
        # IK parent rotations occur around its own reference pivot.
        qz = (0,0,math.sin(.15),math.cos(.15))
        path = vpd_file(folder, [("センター",(0,-2,0),identity),("左足IK親",(0,0,0),qz),
                                 ("左足ＩＫ",(0,0,0),q45)], 'ik_parent.vpd')
        import_mmd_motion(bpy.context,rig,path)
        foot = rig.pose.bones['Bip001_L_Foot']; toe = rig.pose.bones['Bip001_L_Toe0']
        par = np.asarray(Matrix.Rotation(-.3,3,'Y'))
        foot_rest = np.asarray(foot.bone.head_local)
        pivot_offset = reference['foot_from_parent']*scale
        goal = foot_rest-pivot_offset+par@pivot_offset
        np.testing.assert_allclose(foot.head,goal,atol=2e-5)
        toe_vec = np.asarray(toe.bone.head_local-foot.bone.head_local)
        expected_toe = par@waist@toe_vec
        np.testing.assert_allclose(toe.head-foot.head,expected_toe,atol=2e-5)
        # Source-axis twist projection: discard swing instead of preserving its angle.
        axis = reference['directions']['L_arm']; axis /= np.linalg.norm(axis)
        swing_axis = np.cross(axis,[0,0,1]); swing_axis /= np.linalg.norm(swing_axis)
        from mathutils import Quaternion
        twist = Quaternion(Vector(axis),.6); swing = Quaternion(Vector(swing_axis),.8)
        mixed = np.asarray((swing@twist).to_matrix())[None]
        np.testing.assert_allclose(_twist_angle(mixed,axis),[.6],atol=2e-5)
        # An axial skin twist cannot move the elbow or either helper off axis.
        q = np.asarray(twist); mmd_q=np.r_[q[0], -MMD_TO_BLENDER.T@q[1:]]
        path=vpd_file(folder,[("左腕捩",(0,0,0),tuple(mmd_q[[1,2,3,0]]))],'twist.vpd')
        import_mmd_motion(bpy.context,rig,path)
        upper,elbow=(rig.pose.bones['Bip001_L_'+p] for p in ('UpperArm','Forearm'))
        mid=rig.pose.bones['Bip001_LUpArmTwist1']
        np.testing.assert_allclose(mid.head,(upper.head+elbow.head)*.5,atol=2e-5)
        np.testing.assert_allclose(deformation(rig,elbow.name),baseline[elbow.name],atol=2e-5)
        # Quaternion half turns, VMD frame-rate conversion, and Bezier interpolation.
        matrices=np.asarray([Matrix.Rotation(math.pi,3,a) for a in 'XYZ']+[Matrix.Rotation(.8,3,'Y')])
        np.testing.assert_allclose(_qmatrix(_matrix_q(matrices)),matrices,atol=1e-6)
        vmd=importlib.import_module('mmd_tools.core.vmd'); file=vmd.File(); file.boneAnimation=vmd.BoneAnimation()
        for frame,value,quat in [(0,0,identity),(30,1,q45)]:
            k=vmd.BoneFrameKey(); k.frame_number,k.location,k.rotation=frame,(value,0,0),quat; k.interp=[0]*64
            for axis in range(4):
                k.interp[axis],k.interp[axis+4],k.interp[axis+8],k.interp[axis+12]=20,20,107,107
            file.boneAnimation['左腕'].append(k)
        path=str(Path(folder)/'motion.vmd'); file.save(filepath=path); motion=load_motion(path)
        assert len(retarget_motion(rig,motion,fps=24)['frames'])==25
        assert len(retarget_motion(rig,motion,fps=60)['frames'])==61
        track=motion['tracks']['左腕']; loc,quat=sample_track(track,np.array([0.,15,30]))
        np.testing.assert_allclose(loc[:,0],[0,.5,1],atol=1e-5)
        track['interp'][1,[0,4,8,12]]=[0,127,127,127]
        assert sample_track(track,np.array([15.]))[0][0,0] > .85
        # IK toggles disable the entire foot/toe solving stage.
        motion=load_motion(ik_path); motion['toggles']={'左足ik':[(0,False)],'右足ik':[(0,False)]}
        apply_samples(retarget_motion(rig,motion))
        np.testing.assert_allclose(rig.pose.bones['Bip001_L_Foot'].head,[.08,0,.1-2*scale],atol=2e-5)
        # Refuse active constraints before creating/replacing actions.
        current=rig.animation_data.action; actions=len(bpy.data.actions)
        c=rig.pose.bones['Bip001_L_UpperArm'].constraints.new('COPY_ROTATION')
        try:
            import_mmd_motion(bpy.context,rig,path)
            raise AssertionError('Active constraints were accepted')
        except ValueError:
            assert rig.animation_data.action == current and len(bpy.data.actions)==actions
        rig.pose.bones['Bip001_L_UpperArm'].constraints.remove(c)
        assert not any(p.constraints for p in rig.pose.bones)
        print('MMD_ANIMATION_SMOKE_OK: EIEM stance/pivots/folds, FK offsets, foot/toe IK, swing-twist, time, action safety')


if __name__ == '__main__':
    run()
