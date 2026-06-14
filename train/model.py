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

    def forward(
        self,
        spectral: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TerraTorch's Prithvi segmentation models return a ModelOutput or Tensor.
        # The API accepts pixel_values=(B,T,C,H,W) and temporal_coords=(B,T).
        # Adjust here if the installed TerraTorch version uses different arg names.
        out = self.model(
            pixel_values=spectral,
            temporal_coords=temporal_coords,
        )
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
        from terratorch.models import PrithviModelFactory
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

    print(f'[model] Loading Prithvi-EO-2.0-300M via TerraTorch '
          f'(num_frames_max={num_frames_max}, num_classes={num_classes})...')

    # ── TerraTorch model construction ─────────────────────────────────────────
    # If the TerraTorch API changes between versions, adjust only this block.
    # Bands B02,B03,B04,B08,B11,B12 map to Prithvi's Blue,Green,Red,NIR,SWIR1,SWIR2.
    factory = PrithviModelFactory()
    terratorch_model = factory.build_model(
        task='segmentation',
        backbone='prithvi_eo_b_300m',
        decoder='UperNetDecoder',
        in_channels=6,
        num_frames=num_frames_max,
        num_classes=num_classes,
        pretrained=True,
        backbone_kwargs={
            'temporal_coords': True,
            'location_coords': False,
        },
        decoder_kwargs={
            'channels': 256,
        },
    )

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
