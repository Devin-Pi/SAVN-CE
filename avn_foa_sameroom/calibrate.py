"""Independent axis/yaw diagnostic. No per-record truth-dependent sign fixes."""
import argparse,tempfile
from pathlib import Path
import numpy as np
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis,quat_rotate_vector
import trimesh
from .common import native_to_local,local_unit,write_json


def measure(mesh):
    c=hs.SimulatorConfiguration();c.scene_id=str(mesh);c.create_renderer=False
    sim=hs.Simulator(hs.Configuration(c,[hs.agent.AgentConfiguration()]))
    spec=hs.AudioSensorSpec();spec.uuid='calibration';spec.position=[0,0,0];spec.enableMaterials=False
    spec.channelLayout.type=hs.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics;spec.channelLayout.channelCount=4
    ac=spec.acousticsConfig;ac.sampleRate=16000;ac.threadCount=1
    ac.direct=True;ac.indirect=ac.diffraction=ac.transmission=ac.temporalCoherence=False
    sim.add_sensor(spec);sensor=sim.get_agent(0)._sensors[spec.uuid];cases=[]
    try:
        for yaw in (0,np.pi/2,np.pi,3*np.pi/2):
            for direction in np.concatenate((np.eye(3),-np.eye(3),[[1,2,3]/np.sqrt(14)])):
                sensor.reset()
                state=hs.AgentState();state.position=np.zeros(3);state.rotation=quat_from_angle_axis(yaw,np.array([0,1,0]))
                sim.get_agent(0).set_state(state);source=2*direction
                sensor.setAudioSourceTransform(source)
                ir=np.asarray(sim.get_sensor_observations()[spec.uuid],dtype=np.float32).T
                peak=np.linalg.norm(ir,axis=1).argmax();native=ir[peak:peak+1]
                ratio=native_to_local(native,yaw)[0]/native[0,0]
                local=quat_rotate_vector(state.rotation.inverse(),source);unit=np.array([-local[2],-local[0],local[1]])/np.linalg.norm(local);expected=np.array([1,unit[1],unit[2],unit[0]])
                np.testing.assert_allclose(ratio,expected,atol=1e-5,rtol=0)
                assert abs(peak/16000-2/343)<.002
                cases.append(dict(yaw_rad=float(yaw),source_world=source.tolist(),peak_sample=int(peak),
                    native_w_normalized=(native[0]/native[0,0]).tolist(),ambix_w_normalized=ratio.tolist(),expected=expected.tolist(),error=float(np.max(np.abs(ratio-expected)))))
    finally:sim.close()
    return dict(status='passed',cases=cases,count=len(cases),max_error=max(x['error'] for x in cases),
        native_order=['W','+HabitatY','+HabitatZ','+HabitatX'],native_normalization='N3D inferred from axial sqrt(3) / W',
        output_order=['W','Y','Z','X'],output_normalization='SN3D',coordinates='right-handed local +X front +Y left +Z up',
        conversion='uniform mapping, first-order /sqrt(3), signs then inverse receiver yaw rotation',
        expected_basis_method='independent Habitat quaternion inverse rotation, then local [-Z,-X,+Y] axes',diagnostic_receiver_position_m=[0,0,0],dataset_receiver_height_above_base_m=1.5,conversion_contract_version=__import__('spatial_audio.foa',fromlist=['CONTRACT_VERSION']).CONTRACT_VERSION,quaternion_xyzw=True,per_channel_normalization=False,source_distance_m=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory() as d:
        mesh=Path(d)/'diagnostic.glb';box=trimesh.creation.box(extents=[20,20,20]);box.invert();box.export(mesh)
        report=measure(mesh)
    write_json(a.output,report)
    print(report['status'],report['count'],report['max_error'],flush=True)
if __name__=='__main__':main()
