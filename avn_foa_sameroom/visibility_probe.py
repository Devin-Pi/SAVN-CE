"""Isolate sourceIsVisible behavior in the installed binary with known geometry."""
import argparse,tempfile
from pathlib import Path
import numpy as np
import habitat_sim as hs
import trimesh
from .common import write_json
from .generate import audio_spec
from .acoustic_mesh import AcousticMesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);mesh=d/'box.glb';box=trimesh.creation.box(extents=[6,6,6]);box.invert();box.export(mesh)
        c=hs.SimulatorConfiguration();c.scene_id=str(mesh);c.create_renderer=False
        sim=hs.Simulator(hs.Configuration(c,[hs.agent.AgentConfiguration()]));spec=audio_spec();spec.position=[0,0,0]
        spec.acousticsConfig.indirect=spec.acousticsConfig.diffraction=spec.acousticsConfig.transmission=False
        sim.add_sensor(spec);sensor=sim.get_agent(0)._sensors[spec.uuid];cases=[]
        try:
            for z in (-1.,-2.,-2.5):
                sensor.reset();state=hs.AgentState();state.position=np.zeros(3);sim.get_agent(0).set_state(state)
                source=np.array([0.,0.,z]);sensor.setAudioSourceTransform(source)
                ir=np.asarray(sim.get_sensor_observations()[spec.uuid]);am=AcousticMesh(sensor,d/'acoustic.obj')
                clear=am.clear(source,np.zeros(3));assert clear
                cases.append(dict(source=source.tolist(),actual_los=clear,sourceIsVisible=bool(sensor.sourceIsVisible()),direct_rir_peak=float(np.max(np.abs(ir)))))
        finally:sim.close()
    result=dict(cases=cases,installed_api_reliable=all(x['sourceIsVisible']==x['actual_los'] for x in cases),
        method='Known empty 6m box, receiver at center; actual acoustic mesh exported and ray-tested. Reversed ray would hit wall behind source at 2 and 2.5m.',
        workaround='No engine patch. SourceIsVisible is diagnostic only. Require both collision and exported actual acoustic mesh segment LOS.')
    write_json(a.output,result);print(result,flush=True)
if __name__=='__main__':main()
