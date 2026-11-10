"""Load a YAML experiment config and build the model / diffusion objects from it."""
import copy
from typing import Any, Dict

import yaml

from .data import DATASETS
from .diffusion import GaussianDiffusion
from .model import UNET, LegacyUNET
from .schedule import make_schedule


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg['dataset']
    if ds not in DATASETS:
        raise ValueError(f"config dataset '{ds}' not in {list(DATASETS)}")
    cfg.setdefault('paths', {})
    cfg['paths'].setdefault('checkpoint', f'checkpoints/{ds}/ddpm_checkpoint.pt')
    cfg['paths'].setdefault('results', f'results/{ds}')
    return cfg


def apply_overrides(cfg: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """Return a copy of cfg with any non-None override written into cfg['train']."""
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if v is not None:
            cfg['train'][k] = v
    return cfg


def build_model(cfg: Dict[str, Any]):
    m = cfg['model']
    ds = DATASETS[cfg['dataset']]
    arch = m.get('arch', 'unet')
    if arch == 'unet':
        return UNET(in_channels=ds.channels,
                    out_channels=ds.channels,
                    image_size=ds.image_size,
                    base_channels=m['base_channels'],
                    channel_mults=m['channel_mults'],
                    num_res_blocks=m['num_res_blocks'],
                    attention_resolutions=m['attention_resolutions'],
                    num_heads=m['num_heads'],
                    num_groups=m['num_groups'],
                    dropout=m['dropout'],
                    time_embed_dim=m.get('time_embed_dim'))
    if arch == 'legacy':
        return LegacyUNET(channels=m['channels'],
                          attentions=m['attentions'],
                          upscales=m['upscales'],
                          num_groups=m['num_groups'],
                          dropout_prob=m['dropout'],
                          num_heads=m['num_heads'],
                          input_channels=ds.channels,
                          output_channels=ds.channels,
                          time_steps=cfg['diffusion']['timesteps'])
    raise ValueError(f"unknown model.arch '{arch}' (use 'unet' or 'legacy')")


def build_diffusion(cfg: Dict[str, Any], model: UNET) -> GaussianDiffusion:
    d = cfg['diffusion']
    schedule = make_schedule(d['schedule'], d['timesteps'])
    return GaussianDiffusion(model, schedule, variance=d.get('variance', 'beta'))
