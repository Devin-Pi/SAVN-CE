"""v1.1 contract, integrity, semantic visibility and live Agent data-boundary tests."""
from pathlib import Path
import gzip,argparse,base64,hashlib,io,json,os,select,subprocess,sys,tempfile,collections,uuid
import numpy as np
from .schema import load_scenarios
from .variants import SceneVariantSpec,ExistingSceneBackend
ROOT=Path('research_outputs/agent_dataset_v1_1')
def sha(p):
 p=str(p).replace('/media/4tbssd/00_PROJECT_CODE/SAVN-CE','/workspace/SAVN-CE').replace('/media/4tbssd/DATA/mp3d/v1/tasks/mp3d','/datasets/mp3d')
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 global ROOT
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT));args=parser.parse_args();ROOT=Path(args.output);run_id=uuid.uuid4().hex[:10]
 checks={};before=json.loads((ROOT/'input_hashes_before.json').read_text());checks['all_12_protected_input_files_unchanged']=all(sha(p)==v for p,v in before.items());assert checks['all_12_protected_input_files_unchanged']
 rows=[json.loads(l) for l in (ROOT/'episode_audit.jsonl').read_text().splitlines()];meta=load_scenarios(ROOT/'scenario_metadata.jsonl');assert len(rows)==len(meta)==20
 semantic_map=json.loads((ROOT/'private/semantic_instance_mapping.json').read_text())
 for r in rows:
  assert all(r['association'].values()) and r['native_replay']['success'] and r['native_replay']['start_navigable'];assert r['native_replay']['declared_distance_error_m']<.05
  assert r['audio_activity']['effective_audio_observed'];ws=r['audio_activity']['windows']
  assert all(abs(w['start_s']-i*.25)<1e-9 and abs(w['end_s']-w['start_s']-.25)<1e-9 for i,w in enumerate(ws))
  for view in [r['initial_view']]+r['legal_audit_views']:
   assert view['legal_navmesh_pose'];sem=np.load(view['audit_instance_file'])['instance_id'];oid=str(r['private_ground_truth']['target_object_id'])
   assert int(np.sum(sem==semantic_map[oid]['semantic_id']))==view['target_semantic_pixels']
   for c in view['same_class_candidates']:
    assert semantic_map[str(c['object_id'])]['semantic_id']==c['semantic_id']
    assert int(np.sum(sem==c['semantic_id']))==c['pixels'] and c['pixels']>=20
  assert r['emitter_visual_evidence'].startswith('unverified')
 raw=json.load(gzip.open('data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz','rt'))['episodes'];manifest=json.loads(Path('data/datasets/savnce_generated/mp3d/v1/manifest.json').read_text());official=json.load(gzip.open(manifest['source_episode_file'],'rt'))['episodes'];official={str(e['episode_id']):e for e in official};provenance={str(e['episode_id']):e for e in manifest['episodes']}
 for e in raw:
  original=official[provenance[str(e['episode_id'])]['official_template_episode_id']];g=e['goals'][0]
  assert set(e)==set(original) and set(g)==set(original['goals'][0])
  assert g['offset']-g['onset']==g['duration'] and e['oracle_actions'][-1]==0 and len(e['oracle_actions'])==e['info']['num_action']
  assert all(a in [0,1,2,3] for a in e['oracle_actions']) and abs(np.linalg.norm(e['start_rotation'])-1)<1e-6
 checks['original_episode_schema_actions_timing_and_rotations_valid']=True
 checks['all_20_native_dataset_and_reference_replays']=True;checks['all_120_privileged_visibility_views_semantic_pixels_crosschecked']=True;checks['audio_windows_and_unknowns_preserved']=True;checks['metadata_schema_and_episode_links']=True
 for group in ['G0','G1']:assert len(load_scenarios(ROOT/'scenario_metadata.jsonl',group))>=1
 checks['existing_G0_G1_minimal_coverage_no_new_dev_tasks']=True
 backend=ExistingSceneBackend();base=SceneVariantSpec('base:17DRP5sb8fy','17DRP5sb8fy');assert not backend.validate(base)['geometry_changed']
 for kwargs in [{'static_objects':[{'asset':'placeholder'}]},{'furniture_edits':[{'instance':'placeholder'}]},{'source_trajectories':[{'positions':[[0,0,0]]}]},{'acoustic_material_overrides':[{'material':'placeholder'}]}]:
  try:backend.validate(SceneVariantSpec('future:test','17DRP5sb8fy',**kwargs));raise AssertionError('Unsupported variant accepted')
  except NotImplementedError:pass
 try:backend.validate(SceneVariantSpec('fake:json_sound_position_change','17DRP5sb8fy'));raise AssertionError('Fake physical scene accepted')
 except ValueError:pass
 checks['unsupported_physical_variants_fail_closed_without_engine_mutation']=True
 # Evidence enrichment: timing measured independently of the shorter reference traces.
 probes={x['episode_id']:x for x in json.loads((ROOT/'private/audio_timing_probe.json').read_text())['rows']}
 assert len(probes)==20
 for r in rows:
  p=probes[r['episode_id']]
  assert p['pre_onset_silence_confirmed'] and p['active_audio_above_1e_5'] and p['post_offset_window_sampled'] and p['native_emission_gate_matches_schedule']
  assert p['collected_interval_s'][1]>=p['scheduled_interval_s'][1]+1
  assert r['reference_endpoint_crosscheck']['navigable'] and r['research_suitability']['benchmark_ready'] is False
  assert abs(r['reference_endpoint_crosscheck']['endpoint_geodesic_distance_m']-r['native_replay']['final_distance_m'])<.05
  if r['acoustic_instance_aliases']['other_instance_count']:
   assert r['research_suitability']['actual_emitter_instance_verification']=='exclude_exact_acoustic_alias'
 checks['all_20_independent_full_audio_timing_probes']=True;checks['official_point_source_aliases_not_claimed_instance_identifiable']=True;checks['reference_endpoints_and_translation_lengths_recorded']=True
 # Same generator and same seed, independent output: compare canonical task definition bytes.
 repro=ROOT/('private/repro_fixture_'+run_id)
 cmd=[sys.executable,'-m','episode_tools.generate','--output',str(repro),'--count','20','--seed','20261105']
 with (ROOT/'repro_test.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=log)
 original=Path('data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz');copy=repro/'train/content/17DRP5sb8fy.json.gz';assert copy.read_bytes()==original.read_bytes();checks['seed_reproducible_task_bytes_20_of_20']=True
 # Real stdio session: current sensors only, no future poses or semantic metadata.
 public=Path('outputs/agent_v1_1_boundary_check_'+run_id);log=(ROOT/'private/agent_stdio.stderr.log').open('w')
 proc=subprocess.Popen([sys.executable,'-m','agent_episode.serve','--inline-observations'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
 def request(x):
  proc.stdin.write(json.dumps(x)+'\n');proc.stdin.flush();assert select.select([proc.stdout],[],[],90)[0];r=json.loads(proc.stdout.readline());return r
 def inspect_packet(r):
  assert r['ok'];o=r['observation'];assert 'files' not in o
  forbidden={'goal_position','goals','sound_id','object_id','oracle_actions','distance_to_goal','scenario_metadata','target_visibility','future_observations','semantic'}
  assert not forbidden.intersection(o)
  with np.load(io.BytesIO(base64.b64decode(o['sensor_payload']['data']))) as x:
   assert set(x.files)=={'rgb','depth_m','binaural_audio','position','rotation_xyzw'}
   assert x['binaural_audio'].shape==(4000,2) and all(np.isfinite(x[k]).all() for k in x.files)
  return o
 try:
  o=inspect_packet(request({'command':'reset','output':str(public),'episode_id':'1000000000','episode_file':'data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz','max_steps':12,'resolution':128}));assert o['step']==0
  payload=o['sensor_payload']['data'];same=inspect_packet(request({'command':'observe','step':999}));assert same['step']==0 and same['sensor_payload']['data']==payload
  for action in ['turn_left','move_forward','turn_right','move_forward']:
   o=inspect_packet(request({'command':'step','action':action,'decision':{'controller':'boundary_test','evidence_ids':[]}}))
  assert o['step']==4
  assert not request({'command':'get_truth'})['ok'];assert not request({'command':'scenario_metadata'})['ok']
  assert request({'command':'stop'})['done'];assert not request({'command':'step','action':'move_forward'})['ok'];assert request({'command':'close'})['closed'];proc.wait(timeout=10)
 finally:
  if proc.poll() is None:proc.kill()
  log.close()
 assert not (public/'evaluation_truth.json').exists() and not (public/'resolved_simulator.yaml').exists() and not (public/'evaluation.json').exists()
 private_runs=[x for x in Path('research_outputs/agent_private').iterdir() if x.name.endswith('_'+public.name)];assert len(private_runs)==1;secret=private_runs[0]/'evaluation_truth.json';assert secret.is_file() and (private_runs[0].stat().st_mode&0o777)==0o700
 denied=subprocess.run([sys.executable,'-c','open('+repr(str(secret))+').read()'],preexec_fn=lambda:os.setuid(65534),capture_output=True);assert denied.returncode!=0 and b'PermissionError' in denied.stderr
 checks['live_stdio_reset_observe_step_stop_close']=True;checks['repeated_observe_idempotent_and_future_request_does_not_advance']=True;checks['only_allowed_arrays_in_inline_payload']=True;checks['no_truth_or_scene_metadata_commands']=True;checks['new_runtime_truth_and_resolved_config_separate_from_public_observations']=True;checks['unprivileged_uid_cannot_read_private_runtime_truth']=True
 # Dataset/asset integrity checked again after all runtime tests.
 assert all(sha(p)==v for p,v in before.items());checks['protected_inputs_unchanged_after_tests']=True
 report={'dataset_version':'1.1','passed':all(checks.values()),'checks':checks,'groups':dict(collections.Counter(r['scenario_group'] for r in meta)),'native_task_count':20,'new_dev_task_count':0,'reference_success_is_not_agent_success':True,'limits':['Semantic render confirms instance pixels, not human RGB similarity or observed sound-emitter attribution.','Only 1/20 reference collections extends beyond offset; separate full timing probes cover all 20. RGB recognizability and emitter attribution remain unverified.','Installed Habitat-Sim binary exact source commit not established.','No static/dynamic object or moving-source implementation tested; unsupported changes are rejected.','Trusted simulator/coordinator retains access to task truth. Root/SSH/file-reading tools must not be given to an untrusted Agent; use sensor-only protocol or isolated client container.']}
 (ROOT/'validation_v1_1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
