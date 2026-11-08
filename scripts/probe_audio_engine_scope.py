"""Bound sourceIsVisible/query behavior and context reset on the installed build."""
import argparse,hashlib,json,subprocess,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,trimesh,habitat_sim as hs
import habitat_sim._ext.habitat_sim_bindings as binding
from avn_foa_sameroom.acoustic_mesh import AcousticMesh


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def segment_clear(triangles,a,b):
    # Small-fixture independent Moller-Trumbore segment test, no engine query/rtree.
    v=np.asarray(b)-a;length=np.linalg.norm(v);direction=v/length
    e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
    h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.sum(e1*h,axis=1)
    valid=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    offset=np.asarray(a)-triangles[:,0];u=np.sum(offset*h,axis=1)*inv
    q=np.cross(offset,e1);w=np.sum(q*direction,axis=1)*inv;t=np.sum(e2*q,axis=1)*inv
    hit=valid&(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)&(t>1e-5)&(t<length-1e-5)
    return not bool(np.any(hit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[];reset_checks={}
    cpp=Path('/app/habitat-sim/src/esp/sensor/AudioSensor.cpp')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for blocked in (False,True):
            room=trimesh.creation.box(extents=[6,6,6]);room.invert()
            if blocked:
                wall=trimesh.creation.box(extents=[5.8,5.8,.1]);wall.apply_translation([0,0,-.75]);room=trimesh.util.concatenate([room,wall])
            mesh=root/('blocked.glb' if blocked else 'open.glb');room.export(mesh)
            cfg=hs.SimulatorConfiguration();cfg.scene_id=str(mesh);cfg.create_renderer=False
            sim=hs.Simulator(hs.Configuration(cfg,[hs.agent.AgentConfiguration()]))
            spec=hs.AudioSensorSpec();spec.uuid='scope_probe';spec.position=[0,0,0];spec.enableMaterials=False
            spec.channelLayout.type=hs.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics;spec.channelLayout.channelCount=4
            ac=spec.acousticsConfig;ac.sampleRate=16000;ac.threadCount=1;ac.direct=True
            ac.indirect=ac.diffraction=ac.transmission=ac.temporalCoherence=False
            ac.meshSimplification=False
            sim.add_sensor(spec);sensor=sim.get_agent(0)._sensors[spec.uuid]
            try:
                probe_direction=np.array([0.,0.,-1.])
                wall_center=None
                if blocked:
                    sensor.reset();sim.get_agent(0).set_state(hs.AgentState());sensor.setAudioSourceTransform(np.array([0.,0.,-1.]))
                    sim.get_sensor_observations()
                    exported=AcousticMesh(sensor,root/'fixture.obj').mesh
                    inner=exported.vertices[np.isclose(np.abs(exported.vertices),2.9,atol=1e-4).sum(axis=1)>=2]
                    assert len(inner)>=8
                    print("INNER",len(inner),[np.unique(np.round(inner[:,k],4)).tolist() for k in range(3)],flush=True)
                    wall_center=(inner.min(axis=0)+inner.max(axis=0))/2
                    assert .6<np.linalg.norm(wall_center)<.9
                    probe_direction=wall_center/np.linalg.norm(wall_center)
                for z in (-1.,-2.,-2.5):
                    sensor.reset();sim.get_agent(0).set_state(hs.AgentState());source=abs(z)*probe_direction;sensor.setAudioSourceTransform(source)
                    rir=np.asarray(sim.get_sensor_observations()[spec.uuid]).copy()
                    geometry=AcousticMesh(sensor,root/'actual.obj');los=segment_clear(geometry.mesh.triangles,source,np.zeros(3))
                    fixture_matches=los==(not blocked)
                    query=bool(sensor.sourceIsVisible());after=np.asarray(sim.get_sensor_observations()[spec.uuid]).copy()
                    peak=float(np.max(np.abs(rir))) if rir.size else 0.
                    print('FIXTURE',blocked,z,los,peak,geometry.triangles,geometry.bounds,flush=True)
                    assert (peak>1e-9)==los,(blocked,z,peak,los)
                    assert np.array_equal(rir,after),'Query altered direct-only render'
                    cases.append(dict(fixture_wall_center_in_export=wall_center.tolist() if wall_center is not None else None,fixture_direction_chosen_from_geometry_only=True,fixture_expected_matches_export=fixture_matches,acoustic_mesh_bounds=geometry.bounds,acoustic_mesh_triangles=geometry.triangles,blocked=blocked,source=source.tolist(),actual_segment_los=los,sourceIsVisible=query,
                        direct_rir_peak=peak,direct_rir_energy=float(np.sum(rir.astype(float)**2)),query_changed_cached_response=False,
                        direct_render_matches_independent_geometry=True))
                if not blocked:
                    arrays=[]
                    for source in ([0,0,-1],[0,0,2.5],[0,0,-1],[0,0,-1]):
                        sensor.reset();sim.get_agent(0).set_state(hs.AgentState());sensor.setAudioSourceTransform(np.asarray(source))
                        arrays.append(np.asarray(sim.get_sensor_observations()[spec.uuid]).copy())
                    assert np.array_equal(arrays[0],arrays[2]) and np.array_equal(arrays[0],arrays[3])
                    reset_checks=dict(status='passed',sequence='A(-1m Z), B(+2.5m Z), A, fresh-reset A',reset_before_every_render=True,
                        temporalCoherence=False,direct_only=True,A_B_A_max_abs_error=float(np.max(np.abs(arrays[0]-arrays[2]))),
                        scope='Deterministic direct-only state isolation; indirect Monte-Carlo waveforms need not be bitwise identical')
            finally:sim.close()
    text=cpp.read_text();start=text.index('bool AudioSensor:: sourceIsVisible()');end=text.index('\n#endif',start)
    query_body=text[start:end]
    meta=dict(habitat_sim_version=hs.__version__,loaded_binding_path=binding.__file__,loaded_binding_sha256=sha(binding.__file__),
        local_source_tree_git_head=subprocess.check_output(['git','-C','/app/habitat-sim','rev-parse','HEAD'],text=True).strip(),
        AudioSensor_cpp_sha256=sha(cpp),AudioSensor_cpp_path=str(cpp),local_source_query_body=query_body,
        binary_source_match_scope='Local build source plus functional reproduction; no decompilation or inspection inside the prebuilt RLRA_Simulate library',
        renderer_call_chain='SAVNCE -> audio sensor runSimulation -> RLRA_Simulate; sourceIsVisible separately constructs a ray and calls RLRA_TraceRayAnyHit',
        source_query_is_external_binding=True,all_direct_diagnostics_passed=True,
        classification='A supported for this wrapper and tested geometry: external query ray is reversed; no evidence it gates the direct renderer. Internal SDK paths are not exhaustively audited.',
        no_engine_patch=True,no_existing_dataset_rebuilt=True)
    assert any(not c['actual_segment_los'] for c in cases),'No valid blocked fixture; do not conclude render occlusion is verified'
    report=dict(status='passed',engine=meta,cases=cases,wrong_query_cases=sum(c['sourceIsVisible']!=c['actual_segment_los'] for c in cases),reset_test=reset_checks)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='engine'}))

if __name__=='__main__':main()
