"""MoGe-L conditioned FM, retaining the exact D2 DINO-B architecture."""
from src.e_models_20261106 import make_model as legacy_model,ConditionCache as LegacyCache,evaluate as legacy_evaluate
def mapped(cfg):
 value=dict(cfg);assert cfg['stage']=='E2_L';value['stage']='E2_B';return value
def make_model(cfg):return legacy_model(mapped(cfg))
class ConditionCache(LegacyCache):
 def __init__(self,records,capacity=128,cfg=None):super().__init__(records,capacity,mapped(cfg))
def evaluate(model,mode,domains,seeds,limit,device,cfg):return legacy_evaluate(model,mode,domains,seeds,limit,device,mapped(cfg))
