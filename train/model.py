"""
Model loading for Prithvi-EO-2.0-300M fine-tuning via TerraTorch.

Two paths selected by the smoke_test flag:
  - smoke_test=True : returns _SmokeStub (tiny Conv2d) — no TerraTorch required.
                      Exercises the full pipeline on CPU locally.
  - smoke_test=False: loads real Prithvi-EO-2.0-300M via TerraTorch with frozen
                      backbone + UPerNet segmentation head. Requires TerraTorch.
                      Install on GPU VM: pip install terratorch

There is NO silent fallback. If TerraTorch is missing on a non-smoke-test run,
this module raises a clear ImportError with install instructions.
"""

import torch
import torch.nn as nn
from typing import Optional


# ── smoke-test stub ────────────────────────────────────────────────────────────

class _SmokeStub(nn.Module):
    """
    Minimal Conv2d stand-in for --smoke-test runs.

    Accepts the same (spectral, temporal_coords) interface as the Prithvi wrapper
    so the full training/eval pipeline can be exercised on CPU without TerraTorch.

    NOT a fallback for real training — use only with --smoke-test.
    """

    def __init__(self, in_channels: int, num_frames: int, num_classes: int):
        super().__init__()
        # Flatten temporal dimension into channels
        self.conv = nn.Conv2d(in_channels * num_frames, num_classes, kernel_size=1)
        self._num_frames = num_frames
        self._num_classes = num_classes

    def forward(
        self,
        spectral: torch.Tensor,                    # (B, T, C, H, W)
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = spectral.shape
        # Pad or truncate to num_frames if needed (variable T per batch)
        if T < self._num_frames:
            pad = spectral.new_zeros(B, self._num_frames - T, C, H, W)
            spectral = torch.cat([spectral, pad], dim=1)
        elif T > self._num_frames:
            spectral = spectral[:, :self._num_frames]
        return self.conv(spectral.reshape(B, self._num_frames * C, H, W))

    def frozen_encoder_params(self):
        return []

    def head_params(self):
        return list(self.parameters())


# ── Prithvi wrapper ────────────────────────────────────────────────────────────

class PrithviSegWrapper(nn.Module):
    """
    Thin wrapper around TerraTorch's Prithvi-EO-2.0 segmentation model.

    Provides a consistent (spectral, temporal_coords) → logits interface
    regardless of the internal TerraTorch API shape expectations.

    spectral:        (B, T, 6, H, W)  float32, normalized
    temporal_coords: (B, T)           float32, normalized DOY in [0,1]
    returns:         (B, num_classes, H, W)  float32 logits
    """

    def __init__(self, terratorch_model: nn.Module):
        super().__init__()
        self.model = terratorch_model
        self._debug_printed = False

    def _find_backbone(self) -> Optional[torch.nn.Module]:
        """Locate the Prithvi backbone inside the TerraTorch model."""
        for attr in ('backbone', 'encoder', 'model'):
            obj = getattr(self.model, attr, None)
            if obj is not None and hasattr(obj, 'num_frames'):
                return obj
        return None

    def forward(
        self,
        spectral: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = spectral.shape

        # Print actual input shape once so we can verify H=W=128.
        if not self._debug_printed:
            print(f'[model] Input shape to model: (B={B}, T={T}, C={C}, H={H}, W={W})')
            self._debug_printed = True

        # Prithvi's Conv3d patch_embed expects (B, C, T, H, W); our data is (B, T, C, H, W).
        spectral = spectral.permute(0, 2, 1, 3, 4).contiguous()

        # Dynamically update the backbone's temporal dimension so
        # prepare_features_for_image_model reshapes with the actual batch t.
        # BucketSampler guarantees every example in this batch has the same T.
        backbone = self._find_backbone()
        if backbone is not None and hasattr(backbone, 'num_frames'):
            backbone.num_frames = T

        out = self.model(spectral, temporal_coords=temporal_coords)
        # ModelOutput → extract the segmentation logits tensor
        if hasattr(out, 'output'):
            return out.output
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def frozen_encoder_params(self):
        """Parameters belonging to the frozen backbone (for logging)."""
        return [p for n, p in self.model.named_parameters()
                if any(kw in n for kw in ('backbone', 'encoder', 'patch_embed', 'blocks'))
                and not p.requires_grad]

    def head_params(self):
        """Trainable parameters (head only, since backbone is frozen)."""
        return [p for p in self.model.parameters() if p.requires_grad]


# ── factory ────────────────────────────────────────────────────────────────────

def load_model(
    num_frames_max: int,
    num_classes: int = 3,
    device: str = 'cpu',
    smoke_test: bool = False,
    patch_size: int = 128,
) -> nn.Module:
    """
    Load and return the segmentation model.

    Args:
        num_frames_max: Maximum number of temporal frames across all AOIs.
                        Used to size the stub (smoke_test=True) or passed to
                        Prithvi's num_frames parameter (smoke_test=False).
        num_classes:    Number of output classes (3: baseline/grading/built).
        device:         'cpu' or 'cuda'.
        smoke_test:     If True, return _SmokeStub (no TerraTorch needed).

    Returns:
        nn.Module with .forward(spectral, temporal_coords) → (B, C, H, W) logits.
        Also exposes .head_params() for the optimizer.
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
            '\n'
            'TerraTorch is required for real training (not smoke-test).\n'
            '\n'
            'Install on the GPU VM:\n'
            '    pip install terratorch\n'
            '\n'
            'See train/requirements.txt for pinned versions.\n'
            'For local CPU smoke-testing, pass --smoke-test to train.py / evaluate.py.\n'
        )

    print(f'[model] Loading Prithvi-EO-2.0-300M via TerraTorch (num_classes={num_classes})...')
    # Prithvi-EO-2.0 self-describes its bands and temporal handling —
    # do NOT pass in_channels, num_frames, or bands to build_model.

    # ── force-import backbone modules to trigger @register decorators ─────────
    # TerraTorch registers models lazily — the module must be imported before
    # the factory can find the backbone name.  Try every known module path.
    _backbone_module = None
    for _mod in [
        'terratorch.models.backbones.prithvi_eo_v2',
        'terratorch.models.backbones.prithvi_model',
        'terratorch.models.backbones.prithvi',
    ]:
        try:
            import importlib
            _backbone_module = importlib.import_module(_mod)
            print(f'[model] Imported backbone module: {_mod}')
            break
        except ImportError:
            pass

    # ── list everything in the registry for debugging ─────────────────────────
    def _list_registered_backbones():
        # TerraTorch registry
        for reg_path in [
            ('terratorch.registry', 'TERRATORCH_BACKBONE_REGISTRY'),
            ('terratorch.registry', 'MODEL_REGISTRY'),
        ]:
            try:
                mod  = importlib.import_module(reg_path[0])
                reg  = getattr(mod, reg_path[1])
                # Registry objects may expose keys via __iter__, keys(), or _registry
                for getter in [lambda r: list(r), lambda r: list(r.keys()),
                               lambda r: list(r._registry.keys())]:
                    try:
                        all_keys = getter(reg)
                        prithvi  = [k for k in all_keys if 'prithvi' in str(k).lower()]
                        print(f'[model] {reg_path[1]} Prithvi entries: {prithvi}')
                        print(f'[model] {reg_path[1]} ALL entries (first 40): {all_keys[:40]}')
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        # Also list anything the backbone module exports
        if _backbone_module is not None:
            fns = [k for k in dir(_backbone_module)
                   if not k.startswith('_') and 'prithvi' in k.lower()]
            print(f'[model] Backbone module public names: {fns}')

    # ── try factory with candidate backbone names ─────────────────────────────
    _BACKBONE_CANDIDATES = [
        'prithvi_eo_v2_300',
        'prithvi_eo_v2_300m',
        'Prithvi_EO_V2_300M',
    ]

    factory = EncoderDecoderFactory()
    terratorch_model = None

    for backbone_name in _BACKBONE_CANDIDATES:
        try:
            terratorch_model = factory.build_model(
                task='segmentation',
                backbone=backbone_name,
                decoder='UperNetDecoder',
                num_classes=num_classes,
                backbone_kwargs={
                    'pretrained': True,
                    'num_frames': num_frames_max,
                    'img_size': patch_size,
                    'temporal_coords': True,
                    'location_coords': False,
                },
                decoder_kwargs={
                    'channels': 256,
                },
            )
            print(f'[model] Backbone loaded via EncoderDecoderFactory: {backbone_name}')
            break
        except Exception as e:
            print(f'[model] "{backbone_name}" failed: {e}')

    if terratorch_model is None:
        _list_registered_backbones()
        raise RuntimeError(
            'Could not instantiate any Prithvi-EO-2.0-300M backbone.\n'
            'Check the "[model] Backbone module public names" and registry output above\n'
            'and add the correct name as the first entry in _BACKBONE_CANDIDATES in\n'
            'train/model.py.'
        )

    # ── post-init backbone patch ──────────────────────────────────────────────
    # backbone_kwargs may not propagate img_size / num_frames to all TerraTorch
    # versions. Patch attributes directly so prepare_features_for_image_model
    # uses the correct spatial grid. num_frames is also overridden per-batch in
    # PrithviSegWrapper.forward; this sets a sane default.
    _backbone_obj = None
    for _attr in ('backbone', 'encoder'):
        _candidate = getattr(terratorch_model, _attr, None)
        if _candidate is not None and hasattr(_candidate, 'num_frames'):
            _backbone_obj = _candidate
            break

    if _backbone_obj is not None:
        _ph = patch_size // 16   # Prithvi patch size = 16
        _old_nf = getattr(_backbone_obj, 'num_frames', '?')
        _old_pg = getattr(_backbone_obj, 'patch_grid_size', '?')
        if hasattr(_backbone_obj, 'num_frames'):
            _backbone_obj.num_frames = num_frames_max
        if hasattr(_backbone_obj, 'patch_grid_size'):
            _backbone_obj.patch_grid_size = (_ph, _ph)
        print(f'[model] Backbone patched: num_frames {_old_nf}→{num_frames_max}, '
              f'patch_grid_size {_old_pg}→({_ph},{_ph})')
    else:
        print('[model] WARNING: could not locate backbone to patch num_frames/patch_grid_size.')

    # ── freeze backbone / encoder ─────────────────────────────────────────────
    frozen_count = 0
    for name, param in terratorch_model.named_parameters():
        if any(kw in name for kw in ('backbone', 'encoder', 'patch_embed', 'blocks')):
            param.requires_grad_(False)
            frozen_count += 1

    trainable = sum(p.numel() for p in terratorch_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in terratorch_model.parameters())
    print(f'[model] Frozen {frozen_count} param tensors. '
          f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%).')

    model = PrithviSegWrapper(terratorch_model)
    return model.to(torch.device(device))
