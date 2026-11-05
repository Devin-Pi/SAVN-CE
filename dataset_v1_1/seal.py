"""Final read-only input integrity and delivery checks; only writes the new report directory."""
from pathlib import Path
import argparse,hashlib,json

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def mapped(p):return p.replace('/media/4tbssd/00_PROJECT_CODE/SAVN-CE','/workspace/SAVN-CE').replace('/media/4tbssd/DATA/mp3d/v1/tasks/mp3d','/datasets/mp3d')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='research_outputs/agent_dataset_v1_1');args=p.parse_args();root=Path(args.output)
 before=json.loads((root/'input_hashes_before.json').read_text());assert len(before)==12 and all(sha(mapped(k))==v for k,v in before.items())
 docs=['README_DATASET_V1_1_zh-CN.md','CAPABILITY_MATRIX.md','SCENE_VARIANT_DESIGN.md'];results=['episode_audit.jsonl','scenario_metadata.jsonl','coverage_report.md','validation_v1_1.json','TEST_COMMANDS.txt'];assert all(Path(x).is_file() for x in docs) and all((root/x).is_file() for x in results)
 validation=json.loads((root/'validation_v1_1.json').read_text());assert validation['passed'] and all(validation['checks'].values())
 probe=root/'isolated_client_probe.json'
 if probe.exists():
  client=json.loads(probe.read_text());assert client['uid']==65534 and client['no_project_mount'] and all(client['checks'].values());validation['checks']['isolated_client_template_has_no_project_or_dataset_mount']=True
 snapshot=root/'private/snapshot_contract_probe/input_hashes_before.json'
 if snapshot.exists():
  observed=json.loads(snapshot.read_text());assert observed=={mapped(k):v for k,v in before.items()};validation['checks']['fresh_snapshot_command_matches_original_12_input_hashes']=True
 validation['checks']['final_delivery_files_present_and_original_12_inputs_still_unchanged']=True;validation['passed']=all(validation['checks'].values());(root/'validation_v1_1.json').write_text(json.dumps(validation,indent=2)+'\n')
 for name in docs:(root/name).write_bytes(Path(name).read_bytes())
 manifest={'dataset_version':'1.1','code_commit':Path(root/'code_commit.txt').read_text().strip() if (root/'code_commit.txt').exists() else 'unrecorded','privileged':True,'original_inputs_unchanged':True,'delivery_files':{name:sha(root/name) for name in docs+results},'passed_check_count':len(validation['checks'])}
 (root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (root/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(root))+'\n' for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 print('DELIVERY_PASS',len(validation['checks']),'checks; original inputs unchanged',flush=True)
if __name__=='__main__':main()
