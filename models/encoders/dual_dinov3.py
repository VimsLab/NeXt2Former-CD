import inspect
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM

logger = get_logger()


def _strip_prefix(state_dict, prefixes: Sequence[str]) -> dict:
    if not state_dict:
        return state_dict
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_dinov3_model(repo_dir: str, model_name: str) -> nn.Module:
    """
    Load DINOv3 model from local hub without downloading from internet.
    Creates the model architecture without pretrained weights (weights are loaded separately).
    """
    try:
        # Map model names to their hub factory functions
        import sys
        sys.path.insert(0, repo_dir)
        from dinov3.hub import backbones
        
        # Get the factory function for the model
        if not hasattr(backbones, model_name):
            raise AttributeError(f"Model factory '{model_name}' not found in dinov3.hub.backbones")
        
        # Call the factory function with pretrained=False to create model without downloading
        model_factory = getattr(backbones, model_name)
        model = model_factory(pretrained=False)
        
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv3 model '{model_name}' from '{repo_dir}'. "
            "Ensure the DINOv3 repo is cloned locally and the model name matches a factory in that repo."
        ) from exc


def _get_attr_first(obj, names: Sequence[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _normalize_patch_size(patch_size) -> Tuple[int, int]:
    if isinstance(patch_size, (list, tuple)):
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


class DualDINOv3(nn.Module):
    def __init__(
        self,
        model_name: str,
        repo_dir: str,
        pretrained_path: Optional[str] = None,
        out_indices: Optional[Sequence[int]] = None,
        norm_fuse: nn.Module = nn.BatchNorm2d,
        use_frm: bool = True,
        use_adapter: bool = False,
        use_deform_attn: bool = False,
        deform_n_heads: int = 8,
        deform_n_points: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model = _load_dinov3_model(repo_dir, model_name)

        self.embed_dim = _get_attr_first(self.model, ["embed_dim", "dim", "hidden_dim"], None)
        if self.embed_dim is None:
            raise ValueError("Could not infer embed_dim from DINOv3 model.")

        patch_size = _get_attr_first(self.model, ["patch_size"], None)
        if patch_size is None and hasattr(self.model, "patch_embed"):
            patch_size = _get_attr_first(self.model.patch_embed, ["patch_size"], None)
        self.patch_size = _normalize_patch_size(patch_size or 16)

        self.blocks = _get_attr_first(self.model, ["blocks", "layers"], None)
        if self.blocks is not None:
            self.num_blocks = len(self.blocks)
        else:
            self.num_blocks = _get_attr_first(self.model, ["n_blocks", "num_blocks", "depth"], None)
            if self.num_blocks is None and hasattr(self.model, "stages"):
                self.num_blocks = len(self.model.stages)
            if self.num_blocks is None and hasattr(self.model, "downsample_layers"):
                self.num_blocks = len(self.model.downsample_layers)
            if self.num_blocks is None and not hasattr(self.model, "get_intermediate_layers"):
                raise ValueError("Could not find transformer blocks on the DINOv3 model.")

        if out_indices is None:
            if self.num_blocks <= 4:
                out_indices = list(range(self.num_blocks))
            else:
                quarter = max(1, self.num_blocks // 4)
                out_indices = [quarter - 1, 2 * quarter - 1, 3 * quarter - 1, self.num_blocks - 1]
        self.out_indices = list(out_indices)

        self.use_frm = use_frm

        self.is_convnext = hasattr(self.model, "stages") or hasattr(self.model, "downsample_layers")
        self.use_adapter = bool(use_adapter) and (not self.is_convnext)
        stage_dims = None
        if hasattr(self.model, "embed_dims"):
            stage_dims = list(self.model.embed_dims)
        elif hasattr(self.model, "dims"):
            stage_dims = list(self.model.dims)

        if self.use_adapter:
            if len(self.out_indices) != 4:
                raise ValueError("DINOv3 adapter expects 4 interaction indexes for Mask2Former-style decoding.")
            self.stage_dims = [self.embed_dim] * 4
        else:
            if stage_dims is not None and len(stage_dims) > 0:
                self.stage_dims = [stage_dims[i] for i in self.out_indices]
            else:
                self.stage_dims = [self.embed_dim] * len(self.out_indices)

        self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
        self.FFMs = nn.ModuleList(
            [
                FFM(
                    dim=dim,
                    reduction=1,
                    norm_layer=norm_fuse,
                    use_deform_attn=use_deform_attn,
                    deform_n_heads=deform_n_heads,
                    deform_n_points=deform_n_points,
                )
                for dim in self.stage_dims
            ]
        )

        self.out_channels = list(self.stage_dims)

        if self.use_adapter:
            from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
            self.adapter = DINOv3_Adapter(
                self.model,
                interaction_indexes=self.out_indices,
                freeze_backbone=freeze_backbone,
            )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path: str) -> None:
        logger.info(f"Loading DINOv3 weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ["model", "state_dict", "teacher", "student"]:
                if key in ckpt:
                    ckpt = ckpt[key]
                    break
        if isinstance(ckpt, dict):
            ckpt = _strip_prefix(ckpt, ["model.", "backbone.", "student.", "student.backbone."])
        missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.info(f"Missing keys when loading DINOv3: {len(missing)}")
        if unexpected:
            logger.info(f"Unexpected keys when loading DINOv3: {len(unexpected)}")

    def _reshape_tokens(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # tokens: B, N, C
        if tokens.dim() == 4:
            return tokens
        if tokens.shape[1] == H * W + 1:
            tokens = tokens[:, 1:, :]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, H, W).contiguous()

    def _get_intermediate_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Prefer built-in helper if available
        if hasattr(self.model, "get_intermediate_layers"):
            getter = self.model.get_intermediate_layers
            sig = inspect.signature(getter)
            kwargs = {}
            if "reshape" in sig.parameters:
                kwargs["reshape"] = True
            if "return_class_token" in sig.parameters:
                kwargs["return_class_token"] = False
            try:
                return getter(x, self.out_indices, **kwargs)
            except Exception:
                try:
                    return getter(x, n=len(self.out_indices), **kwargs)
                except Exception:
                    return getter(x)

        # Fallback: use hooks on transformer blocks
        if self.blocks is None:
            raise ValueError("DINOv3 model does not expose blocks and get_intermediate_layers failed.")
        feats = {}
        hooks = []

        def _make_hook(idx):
            def _hook(_, __, output):
                feats[idx] = output
            return _hook

        for idx in self.out_indices:
            hooks.append(self.blocks[idx].register_forward_hook(_make_hook(idx)))

        if hasattr(self.model, "forward_features"):
            _ = self.model.forward_features(x)
        else:
            _ = self.model(x)

        for h in hooks:
            h.remove()

        return [feats[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor, x_d: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.use_adapter:
            feats_a = self.adapter(x)
            feats_b = self.adapter(x_d)
            outs: List[torch.Tensor] = []
            for i, key in enumerate(["1", "2", "3", "4"]):
                fa, fb = feats_a[key], feats_b[key]
                if self.use_frm:
                    fa, fb = self.FRMs[i](fa, fb)
                outs.append(self.FFMs[i](fa, fb))
            return tuple(outs)

        H = x.shape[2] // self.patch_size[0]
        W = x.shape[3] // self.patch_size[1]

        feats_a = self._get_intermediate_layers(x)
        feats_b = self._get_intermediate_layers(x_d)

        outs: List[torch.Tensor] = []
        for i, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            fa = self._reshape_tokens(fa, H, W)
            fb = self._reshape_tokens(fb, H, W)
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            outs.append(self.FFMs[i](fa, fb))

        return tuple(outs)