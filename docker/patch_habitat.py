from pathlib import Path
p=Path('/opt/conda/envs/ss/lib/python3.9/site-packages/habitat_sim/simulator.py')
s=p.read_text();s=s.replace('audio_sensor = self._agent._sensors["audio_sensor"]','audio_sensor = self._agent._sensors[self._spec.uuid]');p.write_text(s)
