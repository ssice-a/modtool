"""Non-destructive integration probe on a copy of a live game's armature."""

import importlib
import math
from pathlib import Path
import tempfile
import time

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from modtool.core import mmd_animation as animation
from modtool.core.mmd_reference import tda_reference


def run_live_validation(source_rig):
    scene = bpy.context.scene
    frame, end = scene.frame_current, scene.frame_end
    selected, active = list(bpy.context.selected_objects), bpy.context.view_layer.objects.active
    matrices = {b.name: np.asarray(b.matrix).copy() for b in source_rig.pose.bones}
    action_before = source_rig.animation_data.action if source_rig.animation_data else None
    actions_before = set(bpy.data.actions)
    clone = source_rig.copy()
    clone.data = source_rig.data.copy()
    clone.animation_data_clear()
    clone.parent = None
    clone.name = 'MDT_EIEM_neutral_validation'
    bpy.context.collection.objects.link(clone)
    rest = {b.name: np.asarray(b.matrix_local).copy() for b in clone.data.bones}
    identity = (0., 0., 0., 1.)
    report = {}
    reference = tda_reference()
    vpd = importlib.import_module(animation._mmd_module()+'.core.vpd')
    try:
        with tempfile.TemporaryDirectory(prefix='mdt_eiem_live_') as folder:
            def bake(entries, name):
                file = vpd.File(); file.osm_name='NeutralProbe.osm'
                file.bones = [vpd.VpdBone(n, p, q) for n,p,q in entries]
                path = str(Path(folder)/(name+'.vpd'))
                file.save(filepath=path)
                return animation.import_mmd_motion(bpy.context,clone,path)
            start=time.perf_counter()
            result=bake([('下半身',(0,0,0),identity)],'neutral')
            report['mapped_roles']=len(result['mapping'])
            report['baked_bones']=len(result['samples'])
            report['translation_scale']=result['translation_scale']
            report['neutral_bake_ms']=round((time.perf_counter()-start)*1000,2)
            mapping=result['mapping']
            def head(role):
                return np.asarray(clone.pose.bones[mapping[role]].head)
            def direction(a,b):
                v=head(b)-head(a)
                return v/np.linalg.norm(v)
            def error(a,b):
                return math.degrees(math.acos(float(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
            report['neutral_arm_axis_error_degrees']={s:error(direction(f'{s}_arm',f'{s}_elbow'),reference['directions'][f'{s}_arm']) for s in ('L','R')}
            report['neutral_palm_axis_error_degrees']={s:error(direction(f'{s}_hand',f'{s}_finger2_0'),reference['wrists'][s][0]) for s in ('L','R')}
            report['neutral_palm_roll_error_degrees']={}
            for side in ('L','R'):
                forward=direction(f'{side}_hand',f'{side}_finger2_0')
                lateral=head(f'{side}_finger1_0')-head(f'{side}_finger4_0')
                lateral-=forward*np.dot(lateral,forward)
                sf,sl=reference['wrists'][side]
                sf=sf/np.linalg.norm(sf); sl=sl-sf*np.dot(sl,sf)
                report['neutral_palm_roll_error_degrees'][side]=error(lateral,sl)
            q=(0.,0.,math.sin(.6),math.cos(.6))
            bake([('左ひじ',(0,0,0),q),('右ひじ',(0,0,0),(0.,0.,-q[2],q[3]))],'elbow_bend')
            # Oracle uses MMD XYZ -> Blender XZY with reflected rotation sign.
            from mathutils import Matrix
            report['bent_forearm_axis_error_degrees']={}
            length_error=0.
            for side,sign in (('L',1),('R',-1)):
                expected=np.asarray(Matrix.Rotation(-1.2*sign,3,'Y'))@reference['directions'][f'{side}_elbow']
                report['bent_forearm_axis_error_degrees'][side]=error(direction(f'{side}_elbow',f'{side}_hand'),expected)
                for a,b in (('arm','elbow'),('elbow','hand'),('thigh','knee'),('knee','foot'),('foot','toe')):
                    ar,br=f'{side}_{a}',f'{side}_{b}'
                    original=np.linalg.norm(rest[mapping[br]][:3,3]-rest[mapping[ar]][:3,3])
                    actual=np.linalg.norm(head(br)-head(ar))
                    length_error=max(length_error,abs(original-actual))
            report['max_segment_length_error']=float(length_error)
            scale=result['translation_scale']
            bake([('センター',(0,-1.5,0),identity),('左足ＩＫ',(0,0,0),identity),('右足ＩＫ',(0,0,0),identity)],'knee_bend')
            report['planted_foot_error']={s:float(np.linalg.norm(head(f'{s}_foot')-rest[mapping[f'{s}_foot']][:3,3])) for s in ('L','R')}
            report['toe_direction_error_degrees']={s:error(direction(f'{s}_foot',f'{s}_toe'),rest[mapping[f'{s}_toe']][:3,3]-rest[mapping[f'{s}_foot']][:3,3]) for s in ('L','R')}
            axis=reference['directions']['L_arm']; axis=axis/np.linalg.norm(axis)
            mmd_axis=np.array([axis[0],axis[2],axis[1]])
            q=tuple(-mmd_axis*math.sin(.4))+(math.cos(.4),)
            bake([('左腕捩',(0,0,0),q)],'skin_twist')
            mid=clone.pose.bones['Bip001_LUpArmTwist1']
            a=head('L_arm'); b=head('L_elbow')
            projection=a+(b-a)*np.dot(np.asarray(mid.head)-a,b-a)/np.dot(b-a,b-a)
            report['twist_helper_axis_error']=float(np.linalg.norm(np.asarray(mid.head)-projection))
            report['rest_unchanged']=all(np.array_equal(np.asarray(b.matrix_local),rest[b.name]) for b in clone.data.bones)
            assert max(report['neutral_arm_axis_error_degrees'].values()) < .05, report
            assert max(report['neutral_palm_axis_error_degrees'].values()) < .05, report
            assert max(report['neutral_palm_roll_error_degrees'].values()) < .05, report
            assert max(report['bent_forearm_axis_error_degrees'].values()) < .05, report
            assert length_error < 2e-5, report
            assert max(report['planted_foot_error'].values()) < 2e-5, report
            assert max(report['toe_direction_error_degrees'].values()) < .05, report
            assert report['twist_helper_axis_error'] < 2e-5 and report['rest_unchanged'], report
    finally:
        data=clone.data
        bpy.data.objects.remove(clone,do_unlink=True)
        if data.users==0: bpy.data.armatures.remove(data)
        for action in set(bpy.data.actions)-actions_before:
            bpy.data.actions.remove(action)
        scene.frame_end=end
        scene.frame_set(frame)
        bpy.context.view_layer.objects.active=active
        for obj in bpy.context.selected_objects: obj.select_set(False)
        for obj in selected: obj.select_set(True)
        assert (source_rig.animation_data.action if source_rig.animation_data else None)==action_before
        unchanged=max(float(np.max(np.abs(np.asarray(b.matrix)-matrices[b.name]))) for b in source_rig.pose.bones)
        report['user_pose_restore_error']=unchanged
        assert unchanged < 2e-5, report
    return report
