import os
import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_dinov3_repo_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dinov3_repo = os.path.join(repo_root, "dinov3")
    if dinov3_repo not in sys.path:
        sys.path.insert(0, dinov3_repo)
    # Purge any existing dinov3 modules to avoid resolving to a different installation
    for name in list(sys.modules.keys()):
        if name == "dinov3" or name.startswith("dinov3."):
            del sys.modules[name]


_ensure_dinov3_repo_on_path()
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead


class Mask2FormerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=256,
        input_strides=(4, 8, 16, 32),
        ignore_value=255,
        output_mode="prob",
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Mask2FormerDecoder expects 4 input feature maps.")
        if len(input_strides) != 4:
            raise ValueError("Mask2FormerDecoder expects 4 input strides.")

        input_shape = {
            "1": [in_channels[0], 0, 0, input_strides[0]],
            "2": [in_channels[1], 0, 0, input_strides[1]],
            "3": [in_channels[2], 0, 0, input_strides[2]],
            "4": [in_channels[3], 0, 0, input_strides[3]],
        }

        self.head = Mask2FormerHead(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=ignore_value,
        )
        if output_mode not in {"prob", "logits", "raw"}:
            raise ValueError("Mask2FormerDecoder output_mode must be one of: 'prob', 'logits', 'raw'.")
        self.output_mode = output_mode

    @staticmethod
    def _as_feature_dict(inputs):
        if isinstance(inputs, dict):
            return inputs
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 4:
                raise ValueError("Mask2FormerDecoder expects 4 feature maps.")
            return {"1": inputs[0], "2": inputs[1], "3": inputs[2], "4": inputs[3]}
        raise TypeError("Mask2FormerDecoder expects a dict or tuple/list of 4 feature maps.")

    def _semantic_logits(self, outputs, logits=True):
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]    # [B, Q, H, W]

        if logits:
            class_log_probs = F.log_softmax(pred_logits, dim=-1)[..., :-1]
            mask_log_probs = F.logsigmoid(pred_masks)
            return torch.logsumexp(
                class_log_probs[:, :, :, None, None] + mask_log_probs[:, :, None, :, :], dim=1
            )

        mask_cls = F.softmax(pred_logits, dim=-1)[..., :-1]
        mask_pred = pred_masks.sigmoid()
        return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

    def semantic_logits_from_raw(self, outputs, logits=True):
        return self._semantic_logits(outputs, logits=logits)

    def forward(self, inputs):
        features = self._as_feature_dict(inputs)
        outputs = self.head(features)

        if self.output_mode == "raw":
            return outputs
        if self.output_mode == "logits":
            return self._semantic_logits(outputs, logits=True)
        return self._semantic_logits(outputs, logits=False)
