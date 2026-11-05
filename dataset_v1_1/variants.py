"""Design contract only. Reject unimplemented physical changes rather than silently ignoring them."""
from dataclasses import dataclass,field,asdict
from typing import List,Dict,Any

@dataclass
class SceneVariantSpec:
    scene_variant_id:str
    base_scene_id:str
    dataset_version:str='1.1'
    geometry_epoch:int=0
    static_objects:List[Dict[str,Any]]=field(default_factory=list)
    furniture_edits:List[Dict[str,Any]]=field(default_factory=list)
    source_trajectories:List[Dict[str,Any]]=field(default_factory=list)
    acoustic_material_overrides:List[Dict[str,Any]]=field(default_factory=list)
    validation:Dict[str,str]=field(default_factory=lambda:{k:'unverified' for k in ['visual_geometry','collision_geometry','navmesh','semantic_instances','acoustic_geometry','temporal_audio']})
    def to_dict(self):return asdict(self)

class ExistingSceneBackend:
    """The only implemented backend: the unchanged scanned scene."""
    def validate(self,spec):
        if spec.static_objects or spec.furniture_edits or spec.source_trajectories or spec.acoustic_material_overrides or spec.geometry_epoch!=0:
            raise NotImplementedError('Physical scene variants are design-only in v1.1; refusing to label unchanged rendering as a new physical scene')
        if spec.scene_variant_id!='base:'+spec.base_scene_id:raise ValueError('Base backend requires a canonical base scene ID')
        return {'backend':'existing_scene','geometry_changed':False,'scene_variant_id':spec.scene_variant_id}
    def apply_before_render(self,sim,spec):return self.validate(spec)
