import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead


class Mask2FormerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=2048,
        base_stride=4,
        feature_strides=None,
    ):
        super().__init__()
        if feature_strides is None:
            feature_strides = [base_stride * (2**i) for i in range(len(in_channels))]
        if len(feature_strides) != len(in_channels):
            raise ValueError(
                f"feature_strides length ({len(feature_strides)}) must match in_channels length ({len(in_channels)})."
            )

        input_shape = {
            str(i + 1): [int(in_channels[i]), 0, 0, int(feature_strides[i])] for i in range(len(in_channels))
        }
        self.head = Mask2FormerHead(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )
        self.num_classes = num_classes

    def forward(self, inputs):
        if len(inputs) != 4:
            raise ValueError(f"Mask2FormerDecoder expects 4 feature maps, got {len(inputs)}.")

        features = {str(i + 1): feat for i, feat in enumerate(inputs)}
        outputs = self.head(features)
        return self._semantic_logits(outputs)

    def _semantic_logits(self, outputs):
        pred_logits = outputs["pred_logits"]  # B, Q, C+1
        pred_masks = outputs["pred_masks"]  # B, Q, H, W

        class_log_probs = F.log_softmax(pred_logits, dim=-1)[..., :-1]
        mask_log_probs = F.logsigmoid(pred_masks)

        logits = torch.logsumexp(
            class_log_probs[:, :, :, None, None] + mask_log_probs[:, :, None, :, :], dim=1
        )
        return logits
