"""
Model loading for Prithvi-EO-2.0-300M fine-tuning via TerraTorch.

Two paths selected by the smoke_test flag:
  - smoke_test=True : returns _SmokeStub (tiny Conv2d) — no TerraTorch required.
  - smoke_test=False: loads real Prithvi-EO-2.0-300M via TerraTorch with frozen
                      backbone + UPerNet segmentation head. Requires TerraTorch.

Fixed K=6 window means Prithvi always receives (B, 6, K, H, W) — no variable-length
patching, no monkey-patching of prepare_features_for_image_model.
"""

import torch
import torch.nn as nn
from typing import Optional

from train.dataset import K   # K=6, the fixed causal window length


# ── smoke-test stub ────────────────────────────────────────────────────────────

class _SmokeStub(nn.Module):
    """
    Minimal Conv2d stand-in for --smoke-test runs.
    Same (spectral, temporal_coords) → logits interface as PrithviSegWrapper.
    NOT a fallback for real training.
    """

    def __init__(self, in_channels: int, num_frames: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * num_frames, num_classes, kernel_size=1)
        self._num_frames = num_frames

    def forward(
        self,
        spectral: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = spectral.shape
        return self.conv(spectral.reshape(B, T * C, H, W))

    def frozen_encoder_params(self):
        return []

    def head_params(self):
        return list(self.parameters())


# ── Prithvi wrapper ────────────────────────────────────────────────────────────

class PrithviSegWrapper(nn.Module):
    """
    Thin wrapper around TerraTorch's Prithvi-EO-2.0 segmentation model.

    spectral:        (B, K, 6, H, W)  float32, normalized — fixed K frames
    temporal_coords: (B, K)           float32, normalized DOY in [0, 1]
    returns:         (B, num_classes, H, W)  float32 logits
    """

    def __init__(self, terratorch_model: nn.Module):
        super().__init__()
        self.model = terratorch_model
        self._debug_printed = False

    def forward(
        self,
        spectral: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = spectral.shape

        if not self._debug_printed:
            print(f'[model] Input shape to model: (B={B}, T={T}, C={C}, H={H}, W={W})')
            self._debug_printed = True

        # Prithvi's Conv3d patch_embed expects (B, C, T, H, W).
        spectral = spectral.permute(0, 2, 1, 3, 4).contiguous()

        # backbone_coords_encoding=[] means no temporal/location metadata expected.
        out = self.model(spectral)

        if hasattr(out, 'output'):
            return out.output
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def frozen_encoder_params(self):
        return [p for n, p in self.model.named_parameters()
                if any(kw in n for kw in ('backbone', 'encoder', 'patch_embed', 'blocks'))
                and not p.requires_grad]

    def head_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]


# ── factory ────────────────────────────────────────────────────────────────────

def load_model(
    num_frames_max: int = K,
    num_classes: int = 3,
    device: str = 'cpu',
    smoke_test: bool = False,
    patch_size: int = 128,
) -> nn.Module:
    """
    Load and return the segmentation model.

    Args:
        num_frames_max: Temporal window depth. For production use K=6 (the fixed
                        window constant). Used to size the stub or configure Prithvi.
        num_classes:    Output classes (3: baseline / grading / built).
        device:         'cpu' or 'cuda'.
        smoke_test:     Return _SmokeStub; no TerraTorch needed.
        patch_size:     Spatial patch size in pixels (128 → 8×8 ViT grid with patch=16).
    """
    if smoke_test:
        print('[model] --smoke-test: using _SmokeStub (Conv2d placeholder, no TerraTorch).')
        model = _SmokeStub(in_channels=6, num_frames=num_frames_max, num_classes=num_classes)
        return model.to(torch.device(device))

    # ── real path: requires TerraTorch ────────────────────────────────────────
    try:
        from terratorch.models import EncoderDecoderFactory
    except ImportError:
        raise ImportError(
            '\nTerraTorch is required for real training.\n'
            'Install on the GPU VM:\n'
            '    pip install terratorch\n'
            'For local CPU smoke-testing, pass --smoke-test to train.py / evaluate.py.\n'
        )

    print(f'[model] Loading Prithvi-EO-2.0-300M via TerraTorch '
          f'(num_frames={num_frames_max}, patch_size={patch_size}, num_classes={num_classes})...')

    # ── force-import backbone modules to trigger @register decorators ─────────
    import importlib
    for _mod in [
        'terratorch.models.backbones.prithvi_eo_v2',
        'terratorch.models.backbones.prithvi_model',
        'terratorch.models.backbones.prithvi',
    ]:
        try:
            importlib.import_module(_mod)
            print(f'[model] Imported backbone module: {_mod}')
            break
        except ImportError:
            pass

    # ── registry listing (only on failure) ───────────────────────────────────
    def _list_registered_backbones():
        for reg_path in [
            ('terratorch.registry', 'TERRATORCH_BACKBONE_REGISTRY'),
            ('terratorch.registry', 'MODEL_REGISTRY'),
        ]:
            try:
                mod = importlib.import_module(reg_path[0])
                reg = getattr(mod, reg_path[1])
                for getter in [lambda r: list(r), lambda r: list(r.keys()),
                               lambda r: list(r._registry.keys())]:
                    try:
                        keys = getter(reg)
                        prithvi = [k for k in keys if 'prithvi' in str(k).lower()]
                        print(f'[model] {reg_path[1]} Prithvi entries: {prithvi}')
                        break
                    except Exception:
                        continue
            except Exception:
                pass

    # ── build model with documented flat TerraTorch kwargs ───────────────────
    _BACKBONE_CANDIDATES = [
        'prithvi_eo_v2_300',
        'prithvi_eo_v2_300m',
        'Prithvi_EO_V2_300M',
    ]

    factory          = EncoderDecoderFactory()
    terratorch_model = None

    for backbone_name in _BACKBONE_CANDIDATES:
        try:
            build_kwargs = dict(
                task='segmentation',
                backbone=backbone_name,
                backbone_pretrained=True,
                backbone_num_frames=num_frames_max,   # K=6
                backbone_bands=['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2'],
                backbone_coords_encoding=[],          # no time/loc metadata
                necks=[
                    {'name': 'SelectIndices', 'indices': [5, 11, 17, 23]},
                    {'name': 'ReshapeTokensToImage', 'effective_time_dim': num_frames_max},
                    {'name': 'LearnedInterpolateToPyramidal'},
                ],
                decoder='UNetDecoder',
                decoder_channels=[512, 256, 128, 64],
                num_classes=num_classes,
                head_dropout=0.1,
            )
            terratorch_model = factory.build_model(**build_kwargs)
            print(f'[model] Backbone loaded: {backbone_name}')
            break
        except Exception as e:
            print(f'[model] "{backbone_name}" failed: {e}')

    if terratorch_model is None:
        _list_registered_backbones()
        raise RuntimeError(
            'Could not instantiate any Prithvi-EO-2.0-300M backbone.\n'
            'Check the Prithvi entries printed above and update _BACKBONE_CANDIDATES '
            'in train/model.py.'
        )

    # ── freeze backbone / encoder ─────────────────────────────────────────────
    # Keywords match Prithvi backbone params only. Neck (LearnedInterpolateToPyramidal)
    # and decoder (UNetDecoder) params do NOT match these names and stay trainable.
    _FREEZE_KEYWORDS = ('backbone', 'encoder', 'patch_embed', 'blocks')
    frozen_count = 0
    for name, param in terratorch_model.named_parameters():
        if any(kw in name for kw in _FREEZE_KEYWORDS):
            param.requires_grad_(False)
            frozen_count += 1

    trainable_params = [(n, p) for n, p in terratorch_model.named_parameters()
                        if p.requires_grad]
    trainable = sum(p.numel() for _, p in trainable_params)
    total     = sum(p.numel() for p in terratorch_model.parameters())
    print(f'[model] Frozen {frozen_count} param tensors. '
          f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%).')
    print('[model] Trainable param groups (first 8):')
    for n, p in trainable_params[:8]:
        print(f'  {n}  {tuple(p.shape)}')

    model = PrithviSegWrapper(terratorch_model)
    return model.to(torch.device(device))
