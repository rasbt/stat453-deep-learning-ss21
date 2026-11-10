"""Checkpoint loading, including checkpoints written by the pre-package train.py."""
import os
import re
from typing import Any, Dict

import torch
import torch.nn as nn

from .config import load_config

_LEGACY_KEY = re.compile(r'^(module\.)?Layer(\d+)\.(.*)$')
_LEGACY_ENCODER_LAYERS = 3  # the old UNET had 6 fixed layers: 3 down, 3 up

# Architecture of the single-file train.py era; old checkpoints carry no config.
LEGACY_MODEL_CONFIG = {
    'arch': 'legacy',
    'channels': [64, 128, 256, 512, 512, 384],
    'attentions': [False, True, False, False, False, True],
    'upscales': [False, False, False, True, True, True],
    'num_groups': 32,
    'dropout': 0.0,
    'num_heads': 8,
}


def _remap_legacy_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Layer1..3 -> encoder.0..2, Layer4..6 -> decoder.0..2; keeps any 'module.' prefix."""
    out = {}
    for k, v in sd.items():
        m = _LEGACY_KEY.match(k)
        if m:
            prefix, idx, rest = m.group(1) or '', int(m.group(2)) - 1, m.group(3)
            if idx < _LEGACY_ENCODER_LAYERS:
                k = f'{prefix}encoder.{idx}.{rest}'
            else:
                k = f'{prefix}decoder.{idx - _LEGACY_ENCODER_LAYERS}.{rest}'
        out[k] = v
    return out


def load_checkpoint(path: str, device: torch.device, config_dir: str = 'configs') -> Dict[str, Any]:
    """Load a checkpoint and return it in the current format:
    {'model', 'ema', 'optimizer'?, 'config', 'epoch'}.

    Old checkpoints (keys 'weights'/'ema'/'optimizer', no config) are converted:
    the model keys are renamed and the config is taken from configs/<dataset>.yaml."""
    ck = torch.load(path, map_location=device)
    if 'config' in ck:
        return ck
    dataset = ck.get('dataset', 'mnist')
    cfg = load_config(os.path.join(config_dir, f'{dataset}.yaml'))
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))  # checkpoints/<run>/file -> <run>
    cfg['paths'] = {'checkpoint': path, 'results': os.path.join('results', run)}
    cfg['model'] = dict(LEGACY_MODEL_CONFIG)
    cfg['diffusion'] = {'schedule': 'linear', 'timesteps': 1000, 'variance': 'beta'}
    print(f'Legacy checkpoint detected (dataset={dataset}); using the legacy UNet architecture')
    return {
        'model': _remap_legacy_keys(ck['weights']),
        'ema': _remap_legacy_keys(ck['ema']),
        'optimizer': ck.get('optimizer'),
        'config': cfg,
        'epoch': ck.get('epoch', '?'),
        'legacy': True,
    }


def load_weights(model: nn.Module, ck: Dict[str, Any], use_ema: bool = True) -> None:
    """Load EMA (default) or raw weights into `model`. EMA weights are stored with a
    'module.' prefix by timm's ModelEmaV3, which is stripped here."""
    sd = ck['ema'] if use_ema else ck['model']
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The sinusoidal table is a deterministic buffer; old checkpoints did not store it.
    missing = [k for k in missing if k != 'embeddings.embeddings']
    if missing or unexpected:
        raise RuntimeError(f'state_dict mismatch. missing={missing} unexpected={unexpected}')
