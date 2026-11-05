"""Hash the original episode definitions and stage assets before any audit runs."""
from pathlib import Path
import argparse,json
from .finalize import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='research_outputs/agent_dataset_v1_1');a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True);root.chmod(0o700);dest=root/'input_hashes_before.json'
 if dest.exists():raise FileExistsError('Do not replace the original pre-run hash snapshot')
 files=sorted(x for x in Path('data/datasets/savnce_generated/mp3d/v1').rglob('*') if x.is_file());files+=[Path('audit/official_samples/savnce_dataset/mp3d/v1/train/content/17DRP5sb8fy.json.gz')]
 files += [Path('/datasets/mp3d/17DRP5sb8fy')/x for x in ['17DRP5sb8fy.glb','17DRP5sb8fy.house','17DRP5sb8fy.navmesh','17DRP5sb8fy_semantic.ply']]
 dest.write_text(json.dumps({str(x.resolve()):sha(x) for x in files},indent=2)+'\n');print('SNAPSHOT',len(files))
if __name__=='__main__':main()
