"""Evaluation-side geometry context only; never supplies a direction to the candidate."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from avn_foa_sameroom.plan import make_sim
from avn_foa_sameroom.geometry import Rooms,clear_segment
from avn_foa_sameroom.acoustic_mesh import AcousticMesh
from probe_audio_engine_scope import segment_clear


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    root=a.data/'motion_continuous_seed7'
    meta=json.loads((root/'collection.json').read_text())
    logs=[json.loads(s) for s in (root/'actions.jsonl').read_text().splitlines()]
    source=np.asarray(meta['evaluation_only']['source_position'])
    mesh=Path('/datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb');rooms=Rooms(mesh.with_suffix('.house'))
    sim=make_sim(mesh,7)
    spec=hs.AudioSensorSpec();spec.uuid='direct_geometry_check';spec.position=[0,1.5,0]
    spec.channelLayout.type=hs.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
    spec.channelLayout.channelCount=4;spec.enableMaterials=False
    ac=spec.acousticsConfig;ac.sampleRate=16000;ac.threadCount=4
    ac.indirect=ac.diffraction=ac.transmission=ac.temporalCoherence=False
    sim.add_sensor(spec);sensor=sim.get_agent(0)._sensors[spec.uuid]
    direct=[]
    try:
        distances=[np.linalg.norm(source-r['receiver_position']) for r in logs]
        indices=sorted(set([0,int(np.argmin(distances)),63]))
        for i in indices:
            sensor.reset();state=hs.AgentState();state.position=np.asarray(logs[i]['position'])
            state.rotation=quat_from_coeffs(np.asarray(logs[i]['rotation_xyzw']))
            sim.get_agent(0).set_state(state);sensor.setAudioSourceTransform(source)
            ir=np.asarray(sim.get_sensor_observations()[spec.uuid])
            direct.append(dict(step=i,peak=float(np.abs(ir).max()),energy=float(np.sum(ir.astype(float)**2))))
        geometry=AcousticMesh(sensor,a.data/'diagnostic_acoustic_geometry.obj')
        triangles=geometry.mesh.triangles;lo=triangles.min(axis=1);hi=triangles.max(axis=1)
        rows=[];source_room=rooms.locate(source)
        for i,row in enumerate(logs):
            receiver=np.asarray(row['receiver_position']);boundslo=np.minimum(source,receiver)-.001;boundshi=np.maximum(source,receiver)+.001
            subset=triangles[((hi>=boundslo)&(lo<=boundshi)).all(axis=1)]
            receiver_room=rooms.locate(receiver)
            rows.append(dict(step=i,distance_m=float(distances[i]),receiver_room=receiver_room,
                same_room=(source_room==receiver_room) if source_room is not None and receiver_room is not None else None,
                collision_los=clear_segment(sim,source,receiver),
                acoustic_mesh_los=segment_clear(subset,source,receiver)))
        result=dict(status='completed',scope='offline evaluation context; independent direct-only renderer, actual mesh segments and house floor regions',
            source_room=source_room,source_position=source.tolist(),per_pose=rows,direct_only_renders=direct,
            acoustic_mesh_sha256=geometry.hash,sourceIsVisible_used=False,used_for_model_input=False,
            no_full_wave_acoustic_accuracy_claim=True)
        (a.data/'geometry_context.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='per_pose'}),flush=True)
    finally:sim.close()


if __name__=='__main__':main()
