"""Read actual saved generation settings and native source configuration; no rendering or mutation."""
import argparse,hashlib,inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from savnce.sims.savnce_simulator import SAVNCE_Simulator
import habitat_sim._ext.habitat_sim_bindings as binding


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/workspace/SAVN-CE/research_outputs/AVN-FOA-SameRoom-v0.1')
    cfg=yaml.safe_load((root/'generation_config.yaml').read_text())
    rows=[json.loads(s) for s in (root/'rir_manifest.jsonl').read_text().splitlines()]
    assert cfg['acoustic']['temporalCoherence'] in (False,0) and all(r['history_reset'] for r in rows)
    native=Path(inspect.getfile(SAVNCE_Simulator));text=native.read_text()
    assert 'audio_sensor_spec.acousticsConfig.temporalCoherence = True' in text
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(status='passed',independent_rir_count=len(rows),saved_generation_temporalCoherence=False,
        all_rir_manifest_records_history_reset=True,generator_reset_code='generate.py calls AudioSensor.reset before every independent RIR; C++ reset destroys RLRA context',
        original_config_sha256=sha(root/'generation_config.yaml'),rir_manifest_sha256=sha(root/'rir_manifest.jsonl'),
        native_online_temporalCoherence=True,native_source_path=str(native),native_source_sha256=sha(native),
        loaded_binding_sha256=sha(Path(binding.__file__)),
        interpretation='Independent offline configurations use false plus reset. Native online runs retain true for continuous trajectories; rapid motion can spatially blur IR. These are distinct settings, not proof of dynamic acoustic accuracy.',
        existing_datasets_modified=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
