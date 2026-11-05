import json
from pathlib import Path
REQUIRED={'episode_id','scenario_group','scene_variant_id','target_visibility','visible_same_class_candidates','audio_activity_status','visual_ambiguity','verification_opportunity','annotation_method','annotation_status','dataset_version'}
def load_scenarios(path,group=None):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids=set()
    for r in rows:
        if not REQUIRED<=r.keys() or r['dataset_version']!='1.1':raise ValueError('Invalid v1.1 metadata')
        if r['episode_id'] in ids:raise ValueError('Duplicate episode ID')
        ids.add(r['episode_id'])
        if r['access_class']!='researcher_only_privileged_not_agent_input':raise ValueError('Metadata is privileged')
    return [r for r in rows if group is None or r['scenario_group']==group]
