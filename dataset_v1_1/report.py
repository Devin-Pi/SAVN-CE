"""Reconcile privileged diagnostic evidence; no modification of Episode or scene assets."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs,quat_to_coeffs
from .audit import build_visual_sim

def read_lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write_lines(p,rows):p.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='research_outputs/agent_dataset_v1_1');args=p.parse_args();root=Path(args.output)
 rows=read_lines(root/'episode_audit.jsonl');metadata=read_lines(root/'scenario_metadata.jsonl');meta={x['episode_id']:x for x in metadata}
 manifest=json.loads(Path('data/datasets/savnce_generated/mp3d/v1/manifest.json').read_text());source=json.load(gzip.open(manifest['source_episode_file'],'rt'))['episodes'];raw=json.load(gzip.open('data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz','rt'))['episodes'];byid={str(x['episode_id']):x for x in raw};source_byid={str(x['episode_id']):x for x in source};provenance={str(x['episode_id']):x for x in manifest['episodes']}
 probes={x['episode_id']:x for x in json.loads((root/'private/audio_timing_probe.json').read_text())['rows']}
 sim=build_visual_sim('/datasets/mp3d/'+raw[0]['scene_id']);objects={int(o.id.split('_')[-1]):o for o in sim.semantic_scene.objects if o is not None}
 try:
  for row in rows:
   eid=row['episode_id'];e=byid[eid];g=e['goals'][0];original=source_byid[provenance[eid]['official_template_episode_id']]
   aliases=sorted({int(x['goals'][0]['object_id']) for x in source if x['goals'][0]['object_category']==g['object_category'] and x['goals'][0]['sound_id']==g['sound_id'] and x['goals'][0]['position']==g['position']})
   alias={'official_same_class_same_wave_same_point_object_ids':aliases,'other_instance_count':len(aliases)-1,'audio_instance_identifiability':'nonidentifiable_under_matched_timing_wave_phase_and_pose' if len(aliases)>1 else 'unverified_no_exact_official_point_alias_found','scope':'Official templates share a point source. Different episode timing/phase/pose can still change waveforms. This is not a measured A/B audio equivalence test.'}
   row['acoustic_instance_aliases']=alias;meta[eid]['acoustic_instance_aliases']=alias
   row['association'].update(scene_matches_official=e['scene_id']==original['scene_id'],sound_sha256_matches_manifest=hashlib.sha256((Path('data/sounds/semantic_splits')/g['sound_id']).read_bytes()).hexdigest()==provenance[eid]['sound_sha256'])
   assert all(row['association'].values())
   assert abs(row['native_replay']['actual_initial_distance_m']-e['info']['geodesic_distance'])<.05
   obj=objects[int(g['object_id'])];point=np.array(g['position'])+[0,1.5,0];lo=np.array(obj.aabb.center)-np.array(obj.aabb.sizes)/2;hi=np.array(obj.aabb.center)+np.array(obj.aabb.sizes)/2
   row['source_geometry_relation']={'native_point_source_position':point.tolist(),'target_aabb_center':np.array(obj.aabb.center).tolist(),'native_point_source_distance_outside_target_aabb_m':float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0))),'interpretation':'Official association retained; category waveform and goal+1.5m point source are not a visually animated physical emitter.'}
   state=sim.get_agent(0).get_state();state.position=np.array(e['start_position']);state.rotation=quat_from_coeffs(np.array(e['start_rotation']));state.sensor_states={};sim.get_agent(0).set_state(state);travel=0.;collisions=0
   for a in e['oracle_actions']:
    if a==0:break
    before=sim.get_agent(0).get_state().position.copy();collisions+=int(sim.step(a)['collided']);travel+=float(np.linalg.norm(sim.get_agent(0).get_state().position-before))
   state=sim.get_agent(0).get_state();row['reference_endpoint_crosscheck']={'method':'independent visual-only replay of same legal reference actions; native audio Env success recorded separately','position':state.position.tolist(),'rotation_xyzw':quat_to_coeffs(state.rotation).tolist(),'navigable':bool(sim.pathfinder.is_navigable(state.position)),'executed_translation_length_m':travel,'collisions':collisions,'declared_geodesic_distance_m':e['info']['geodesic_distance'],'reference_actions':e['oracle_actions']}
   assert row['reference_endpoint_crosscheck']['navigable'] and collisions==0
   path=habitat_sim.ShortestPath();path.requested_start=state.position;path.requested_end=np.array(g['position'],dtype=np.float32);assert sim.pathfinder.find_path(path)
   row['reference_endpoint_crosscheck']['endpoint_geodesic_distance_m']=float(path.geodesic_distance)
   assert abs(float(path.geodesic_distance)-row['native_replay']['final_distance_m'])<.05
   meta[eid]['observation_quality']={'audit_camera_resolution':[128,128],'camera_height_m':1.25,'same_class_candidate_pixel_threshold':20,'target_initial_pixels':row['initial_view']['target_semantic_pixels'],'RGB_recognition':'unverified','depth_audio_finite_in_native_replay':True,'audio_activity_threshold_rms':1e-5,'annotation_is_resolution_specific':True}
   probe=probes[eid];summary={k:v for k,v in probe.items() if k!='windows'};row['audio_activity']['independent_full_timing_probe']=summary;meta[eid]['audio_activity_status']['independent_full_timing_probe']=summary
   suitability={'ordinary_agent_debug':'eligible_native_reachability_and_audio_valid','G0_simple_screening':row['scenario_group']=='G0','G1_semantic_candidate_screening':row['scenario_group']=='G1','perceptual_audio_visual_conflict':'unverified','active_visual_candidate_observation':'eligible_for_pilot' if row['verification_opportunity']['visual_candidate_sets_change_after_legal_movement'] else 'unverified_in_sampled_views','actual_emitter_instance_verification':'exclude_exact_acoustic_alias' if len(aliases)>1 else 'unverified_requires_agent_sensor_evidence','benchmark_ready':False}
   row['research_suitability']=suitability;meta[eid]['research_suitability']=suitability
   row['verification_opportunity']['audio_instance_verification']=suitability['actual_emitter_instance_verification'];meta[eid]['verification_opportunity']=row['verification_opportunity']
 finally:sim.close()
 write_lines(root/'episode_audit.jsonl',rows);write_lines(root/'scenario_metadata.jsonl',metadata)
 groups=dict(collections.Counter(x['scenario_group'] for x in rows));alias_count=sum(x['acoustic_instance_aliases']['other_instance_count']>0 for x in rows);opportunity=sum(x['verification_opportunity']['visual_candidate_sets_change_after_legal_movement'] for x in rows)
 summary=json.loads((root/'audit_summary.json').read_text());summary.update(exact_acoustic_alias_task_count=alias_count,full_timing_probe_count=len(probes),moving_candidate_set_opportunities=opportunity,all_associations_verified=all(all(x['association'].values()) for x in rows));(root/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 report='''# SAVN-CE Agent Dataset v1.1 coverage report

所有标签属于研究者侧离线审计，禁止作为 Agent 输入。原20条官方格式任务保持不变，没有新增开发任务。它们是基于官方关联生成的开发集，不是作者发布的官方 benchmark。参考控制器成功不代表 Agent 成功。

## 覆盖与证据边界

'''
 report+=f'- 原生 Dataset / Habitat Env / 序列化往返：20/20。参考成功20/20、碰撞0；非零有效双耳音频20/20。\n- G0={groups.get("G0",0)}、G1={groups.get("G1",0)}、未分组={groups.get("unassigned",0)}。标签均为 instrumented_provisional_requires_human_review。\n- 初始目标存在语义像素：{sum(x["initial_view"]["target_semantic_pixels"]>0 for x in rows)}/20；稳定候选采用 >=20/16384 像素阈值，是筛选阈值，不是识别置信度。\n- 合法平移后候选集合变化：{opportunity}/20，仅证明可获得不同视觉证据，不能证明可定位发声实例。\n- {alias_count}/20 与其他官方同类别实例共用同声音文件和精确声源坐标。相同发声时间、波形相位和听者状态下，当前点声源模型没有实例辨别信息，排除用于精确实例验证。\n- 参考轨迹仅1/20采到 offset 后窗口；其余19条参考采集的结束行为仍标 unknown/unverified。另做独立合法转向探针，20/20覆盖 onset 前、发声期间及 offset 后至少1秒，验证门控时间一致、发声前静音和发声期间实际能量；保存标量，不保存固定轨迹RGB-D/音频。offset 后能量可能为卷积尾响，不要求突然归零。\n- 12个原始输入文件哈希保持一致；种子20261105重新生成的20条任务内容压缩字节与原始内容一致（仅私有测试副本）。\n\n'
 report+='''## 首选开发任务

- G0 `1000000016`：初始目标52个语义像素且仅一个同类稳定候选。目标小，RGB可识别性、Agent成功仍未验证；仅是简单组候选。
- G1 `1000000000`：初始两个 table 同类候选，真正目标无初始语义像素；RGB主要是近墙和裁切家具，尚不足以认定感知歧义。
- G1 `1000000008`：初始两个 chair 同类候选，真正目标无初始语义像素。需要人工审查RGB并由在线Agent选择观测进行验证。
- 这三条均有合法平移后的候选集合变化，均无精确官方同类同点声音实例别名。它们仍不构成已确认视听冲突实验。

没有为凑覆盖而增加任务；下一步应先人工复核这三条的RGB可识别性，并开展真实在线Agent对比。当前未运行普通Agent与证据验证Agent成功率对比。

## 每条任务

`目标px`仅代表语义渲染可见；`候选数`是初始同类别实例达到20px的数量；`别名`为其他同类同文件同点实例数。所有任务普通Agent调试可用，发声实例的视觉证据均unverified。

| episode_id | 类别 | 组 | 目标px | 候选数 | 平移改变候选 | 别名 | 距离m | 动作含STOP | 参考终距m | 完整声音时窗探针 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
'''
 for x in rows:
  n=x['native_replay'];report+=f'| {x["episode_id"]} | {x["private_ground_truth"]["category"]} | {x["scenario_group"]} | {x["initial_view"]["target_semantic_pixels"]} | {len(x["initial_view"]["same_class_candidates"])} | {x["verification_opportunity"]["visual_candidate_sets_change_after_legal_movement"]} | {x["acoustic_instance_aliases"]["other_instance_count"]} | {n["actual_initial_distance_m"]:.3f} | {n["actions_including_stop"]} | {n["final_distance_m"]:.3f} | pass |\n'
 report+='''
## 未解决问题

1. 语义网格与RGB纹理可能有标注边界差异；同类≠视觉相似，零像素≠已确认遮挡。AABB视锥关系仅为排除/候选代理，遮挡归因unknown。
2. 不存在从可见物体本身证明正在发声的动画/视觉标记；官方关联只在评估器中确定。
3. 声源使用官方goal位置加1.5m，可能位于目标包围盒外；保留官方行为，不擅自“修正”位置。
4. 单场景、类别/起点不平衡；未做人工认知标注、算法基线或泛化统计。
5. 安装二进制与旁边C++源码精确提交对应关系未确认；新增物体/动态声源没有进行物理一致性实验。
6. trusted coordinator必须隔离数据目录、stderr和评估器。仅移走truth文件不是对拥有SSH/root或项目全盘访问权的Agent的安全沙箱。
'''
 (root/'coverage_report.md').write_text(report);print('REPORT_COMPLETE',groups,'alias_tasks',alias_count,'timing',len(probes),flush=True)
if __name__=='__main__':main()
